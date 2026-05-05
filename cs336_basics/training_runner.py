from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.adam import AdamW, gradient_clipping, learning_rate_cosine_anneal
from cs336_basics.checkpoint_handling import load_checkpoint, save_checkpoint
from cs336_basics.data_loader import get_batch
from cs336_basics.models.cross_entropy_loss import cross_entropy
from cs336_basics.models.transformer_lm import TransformerLM


METRIC_FIELDS = [
    "iteration",
    "epoch",
    "batch",
    "event",
    "split",
    "loss",
    "learning_rate",
    "total_processed_tokens",
    "elapsed_seconds",
]

PERFORMANCE_FIELDS = [
    "iteration",
    "epoch",
    "batch",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "context_length",
    "total_processed_tokens",
    "step_seconds",
    "data_wait_seconds",
    "forward_seconds",
    "backward_seconds",
    "optimizer_seconds",
    "tokens_per_second",
    "device",
    "gpu_utilization_percent",
    "gpu_memory_allocated_gb",
    "gpu_memory_reserved_gb",
    "mps_driver_allocated_gb",
]


def _nested_get(config: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _first_config_value(config: dict[str, Any], *paths: str | tuple[str, ...], default: Any = None) -> Any:
    for path in paths:
        keys = (path,) if isinstance(path, str) else path
        value = _nested_get(config, keys)
        if value is not None:
            return value
    return default


def _require_config_value(config: dict[str, Any], *paths: str | tuple[str, ...]) -> Any:
    value = _first_config_value(config, *paths)
    if value is None:
        choices = [".".join((path,) if isinstance(path, str) else path) for path in paths]
        raise ValueError(f"Missing required config value. Expected one of: {', '.join(choices)}")
    return value


def _resolve_device(configured_device: str | None) -> torch.device:
    if configured_device and configured_device != "auto":
        return torch.device(configured_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize_dtype_name(dtype_name: Any) -> str:
    return str(dtype_name).strip().lower().replace("-", "").replace("_", "")


def _resolve_weight_dtype(config: dict[str, Any]) -> tuple[str, torch.dtype]:
    dtype_name = _first_config_value(
        config,
        ("model", "dtype"),
        ("model", "weight_dtype"),
        "dtype",
        "weight_dtype",
        ("training", "dtype"),
        ("training", "weight_dtype"),
        default="fp32",
    )
    normalized = _normalize_dtype_name(dtype_name)
    supported_dtypes = {
        "fp32": ("fp32", torch.float32),
        "float32": ("fp32", torch.float32),
        "fp16": ("fp16", torch.float16),
        "float16": ("fp16", torch.float16),
        "half": ("fp16", torch.float16),
        "bf16": ("bf16", torch.bfloat16),
        "bfloat16": ("bf16", torch.bfloat16),
    }
    if normalized in supported_dtypes:
        return supported_dtypes[normalized]

    unsupported_float8 = {
        "fp8",
        "float8",
        "float8e4m3fn",
        "e4m3fn",
        "bf8",
        "float8e5m2",
        "e5m2",
    }
    if normalized in unsupported_float8:
        raise ValueError(
            f"Weight dtype {dtype_name!r} is recognized, but this trainer does not support raw float8 "
            "training. Use fp32, bf16, or fp16. FP8/BF8 training needs explicit scaling and "
            "float8-aware kernels rather than simply casting model weights."
        )

    raise ValueError(f"Unsupported weight dtype {dtype_name!r}. Expected one of: fp32, bf16, fp16, fp8, bf8.")


def _load_memmapped_npy(path: str | os.PathLike, name: str, context_length: int) -> np.memmap:
    dataset = np.load(path, mmap_mode="r")
    if dataset.ndim != 1:
        raise ValueError(f"{name} dataset must be a 1D token array, got shape {dataset.shape}")
    if len(dataset) <= context_length:
        raise ValueError(
            f"{name} dataset has {len(dataset)} tokens, but context_length is {context_length}; "
            "it must contain at least context_length + 1 tokens."
        )
    return dataset


def _build_model(config: dict[str, Any]) -> TransformerLM:
    model_config = config.get("model", {})
    return TransformerLM(
        vocab_size=int(_require_config_value(config, ("model", "vocab_size"), "vocab_size")),
        context_length=int(_require_config_value(config, ("model", "context_length"), "context_length")),
        d_model=int(_require_config_value(config, ("model", "d_model"), "d_model")),
        num_layers=int(_require_config_value(config, ("model", "num_layers"), "num_layers")),
        num_heads=int(_require_config_value(config, ("model", "num_heads"), "num_heads")),
        d_ff=int(_require_config_value(config, ("model", "d_ff"), "d_ff")),
        rope_theta=float(model_config.get("rope_theta", config.get("rope_theta", 10000.0))),
    )


def _build_optimizer(config: dict[str, Any], model: torch.nn.Module) -> AdamW:
    optimizer_config = config.get("optimizer", {})
    betas = optimizer_config.get("betas")
    if betas is None:
        beta1 = float(optimizer_config.get("beta1", 0.9))
        beta2 = float(optimizer_config.get("beta2", 0.999))
    else:
        beta1 = float(betas[0])
        beta2 = float(betas[1])
    return AdamW(
        model.parameters(),
        lr=float(optimizer_config.get("lr", config.get("learning_rate", 1e-3))),
        betas=(beta1, beta2),
        eps=float(optimizer_config.get("eps", 1e-8)),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
    )


def _compute_steps_per_epoch(config: dict[str, Any], train_data: np.memmap, batch_size: int, context_length: int) -> int:
    configured = _first_config_value(
        config,
        ("training", "batches_per_epoch"),
        "batches_per_epoch",
        ("training", "steps_per_epoch"),
        "steps_per_epoch",
    )
    if configured is not None:
        return max(1, int(configured))
    return max(1, (len(train_data) - 1) // (batch_size * context_length))


def _compute_max_iterations(config: dict[str, Any], steps_per_epoch: int) -> int:
    configured_iterations = _first_config_value(
        config,
        ("training", "num_iterations"),
        ("training", "num_iterrations"),
        "num_iterations",
        "num_iterrations",
        ("training", "max_iters"),
        "max_iters",
    )
    if configured_iterations is not None:
        return max(0, int(configured_iterations))

    training_config = config.get("training", {})
    num_epochs = int(training_config.get("num_epochs", config.get("num_epochs", 1)))
    return max(0, num_epochs * steps_per_epoch)


def _learning_rate_for_iteration(config: dict[str, Any], iteration: int, max_iterations: int) -> float:
    optimizer_config = config.get("optimizer", {})
    base_lr = float(optimizer_config.get("lr", config.get("learning_rate", 1e-3)))
    max_lr = float(optimizer_config.get("max_learning_rate", base_lr))
    min_lr = float(optimizer_config.get("min_learning_rate", base_lr))
    warmup_iters = int(optimizer_config.get("warmup_iters", 0))
    cosine_cycle_iters = int(optimizer_config.get("cosine_cycle_iters", max_iterations))

    return learning_rate_cosine_anneal(
        iteration,
        max_learning_rate=max_lr,
        min_learning_rate=min_lr,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    )


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _device_performance_stats(device: torch.device) -> dict[str, str]:
    if device.type == "cuda":
        return {
            "gpu_utilization_percent": "",
            "gpu_memory_allocated_gb": f"{torch.cuda.memory_allocated(device) / 1e9:.3f}",
            "gpu_memory_reserved_gb": f"{torch.cuda.memory_reserved(device) / 1e9:.3f}",
            "mps_driver_allocated_gb": "",
        }
    if device.type == "mps":
        return {
            "gpu_utilization_percent": "",
            "gpu_memory_allocated_gb": f"{torch.mps.current_allocated_memory() / 1e9:.3f}",
            "gpu_memory_reserved_gb": "",
            "mps_driver_allocated_gb": f"{torch.mps.driver_allocated_memory() / 1e9:.3f}",
        }
    return {
        "gpu_utilization_percent": "",
        "gpu_memory_allocated_gb": "",
        "gpu_memory_reserved_gb": "",
        "mps_driver_allocated_gb": "",
    }


def _lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    return cross_entropy(logits.float().reshape(-1, vocab_size), targets.reshape(-1))


@torch.no_grad()
def _estimate_loss(
    model: TransformerLM,
    dataset: np.memmap,
    batch_size: int,
    context_length: int,
    device: torch.device,
    eval_iters: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(eval_iters):
        inputs, targets = get_batch(dataset, batch_size, context_length, str(device))
        loss = _lm_loss(model(inputs), targets)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def _append_metric(metrics_path: Path, row: dict[str, Any]) -> None:
    is_new_file = not metrics_path.exists()
    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


def _append_performance(performance_path: Path, row: dict[str, Any]) -> None:
    is_new_file = not performance_path.exists()
    with performance_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PERFORMANCE_FIELDS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


def _read_loss_points(metrics_path: Path) -> dict[str, list[tuple[int, float]]]:
    points: dict[str, list[tuple[int, float]]] = {"train": [], "val": []}
    if not metrics_path.exists():
        return points

    with metrics_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split = row.get("split")
            if split not in points:
                continue
            try:
                points[split].append((int(row["iteration"]), float(row["loss"])))
            except (KeyError, ValueError):
                continue
    return points


def _polyline(points: list[tuple[int, float]], color: str, x_min: int, x_max: int, y_min: float, y_max: float) -> str:
    if not points:
        return ""

    width = 760
    height = 360
    left = 70
    top = 30
    x_span = max(1, x_max - x_min)
    y_span = max(1e-12, y_max - y_min)
    coords = []
    for x_value, y_value in points:
        x = left + (x_value - x_min) / x_span * width
        y = top + height - (y_value - y_min) / y_span * height
        coords.append(f"{x:.2f},{y:.2f}")
    return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(coords)}" />'


def _write_loss_plot(metrics_path: Path, plot_path: Path) -> None:
    points = _read_loss_points(metrics_path)
    all_points = points["train"] + points["val"]
    if not all_points:
        return

    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    y_padding = max(1e-6, (y_max - y_min) * 0.05)
    y_min -= y_padding
    y_max += y_padding

    train_line = _polyline(points["train"], "#2563eb", x_min, x_max, y_min, y_max)
    val_line = _polyline(points["val"], "#dc2626", x_min, x_max, y_min, y_max)
    plot_path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="460" viewBox="0 0 900 460">
  <rect width="900" height="460" fill="white" />
  <line x1="70" y1="390" x2="830" y2="390" stroke="#111827" stroke-width="1" />
  <line x1="70" y1="30" x2="70" y2="390" stroke="#111827" stroke-width="1" />
  <text x="450" y="445" text-anchor="middle" font-family="sans-serif" font-size="14">Iteration</text>
  <text x="20" y="210" text-anchor="middle" font-family="sans-serif" font-size="14"
        transform="rotate(-90 20 210)">Loss</text>
  <text x="70" y="414" font-family="sans-serif" font-size="12">{x_min}</text>
  <text x="830" y="414" text-anchor="end" font-family="sans-serif" font-size="12">{x_max}</text>
  <text x="64" y="394" text-anchor="end" font-family="sans-serif" font-size="12">{y_min:.4f}</text>
  <text x="64" y="34" text-anchor="end" font-family="sans-serif" font-size="12">{y_max:.4f}</text>
  {train_line}
  {val_line}
  <rect x="680" y="42" width="12" height="12" fill="#2563eb" />
  <text x="700" y="53" font-family="sans-serif" font-size="13">train</text>
  <rect x="680" y="66" width="12" height="12" fill="#dc2626" />
  <text x="700" y="77" font-family="sans-serif" font-size="13">validation</text>
</svg>
""",
        encoding="utf-8",
    )


def _log_metric(
    metrics_path: Path,
    plot_path: Path,
    iteration: int,
    epoch: int,
    batch: int,
    event: str,
    split: str,
    loss: float,
    learning_rate: float,
    total_processed_tokens: int,
    start_time: float,
) -> None:
    row = {
        "iteration": iteration,
        "epoch": epoch,
        "batch": batch,
        "event": event,
        "split": split,
        "loss": f"{loss:.8f}",
        "learning_rate": f"{learning_rate:.12g}",
        "total_processed_tokens": total_processed_tokens,
        "elapsed_seconds": f"{time.time() - start_time:.3f}",
    }
    _append_metric(metrics_path, row)
    _write_loss_plot(metrics_path, plot_path)


def _log_performance(
    performance_path: Path,
    iteration: int,
    epoch: int,
    batch: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    context_length: int,
    timings: dict[str, float],
    device: torch.device,
) -> None:
    step_seconds = timings["step"]
    effective_batch_size = batch_size * gradient_accumulation_steps
    total_processed_tokens = iteration * effective_batch_size * context_length
    tokens_per_second = effective_batch_size * context_length / step_seconds if step_seconds > 0 else 0.0
    row = {
        "iteration": iteration,
        "epoch": epoch,
        "batch": batch,
        "micro_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "context_length": context_length,
        "total_processed_tokens": total_processed_tokens,
        "step_seconds": f"{step_seconds:.6f}",
        "data_wait_seconds": f"{timings['data_wait']:.6f}",
        "forward_seconds": f"{timings['forward']:.6f}",
        "backward_seconds": f"{timings['backward']:.6f}",
        "optimizer_seconds": f"{timings['optimizer']:.6f}",
        "tokens_per_second": f"{tokens_per_second:.2f}",
        "device": str(device),
        **_device_performance_stats(device),
    }
    _append_performance(performance_path, row)
    memory_parts = []
    if row["gpu_memory_allocated_gb"]:
        memory_parts.append(f"gpu_mem={row['gpu_memory_allocated_gb']}GB")
    if row["mps_driver_allocated_gb"]:
        memory_parts.append(f"mps_driver={row['mps_driver_allocated_gb']}GB")
    memory_summary = f" {' '.join(memory_parts)}" if memory_parts else ""
    print(
        f"perf iter {iteration}: step={step_seconds:.3f}s data={timings['data_wait']:.3f}s "
        f"fwd={timings['forward']:.3f}s bwd={timings['backward']:.3f}s "
        f"opt={timings['optimizer']:.3f}s tokens/s={tokens_per_second:.0f} "
        f"total_tokens={total_processed_tokens}{memory_summary}"
    )


def _checkpoint_paths(run_dir: Path, iteration: int, epoch: int | None = None) -> list[Path]:
    paths = [run_dir / "checkpoint_latest.pt", run_dir / f"checkpoint_iter_{iteration}.pt"]
    if epoch is not None:
        paths.append(run_dir / f"checkpoint_epoch_{epoch}.pt")
    return paths


def _save_checkpoint_set(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    run_dir: Path,
    epoch: int | None = None,
) -> None:
    for checkpoint_path in _checkpoint_paths(run_dir, iteration, epoch):
        save_checkpoint(model, optimizer, iteration, checkpoint_path)


def _matching_run_dirs(output_dir: Path, config_stem: str) -> list[Path]:
    if not output_dir.exists():
        return []

    exact_suffix = f"-{config_stem}"
    numbered_suffix_marker = f"{exact_suffix}-"
    return sorted(
        [
            path
            for path in output_dir.iterdir()
            if path.is_dir()
            and (
                path.name.endswith(exact_suffix)
                or (
                    numbered_suffix_marker in path.name
                    and path.name.rsplit(numbered_suffix_marker, maxsplit=1)[1].isdigit()
                )
            )
        ],
        key=lambda path: path.name,
    )


def _resolve_run_dir(output_dir: Path, config_stem: str, resume: str | os.PathLike | None) -> Path:
    if resume is not None and str(resume) == "latest":
        candidates = _matching_run_dirs(output_dir, config_stem)
        if candidates:
            return candidates[-1]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"{timestamp}-{config_stem}"
    if not run_dir.exists():
        return run_dir

    suffix = 2
    while True:
        suffixed_run_dir = output_dir / f"{timestamp}-{config_stem}-{suffix}"
        if not suffixed_run_dir.exists():
            return suffixed_run_dir
        suffix += 1


def train(config_path: str | os.PathLike, resume: str | os.PathLike | None = None) -> Path:
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    output_dir = Path(config.get("output_dir", "./logs"))
    run_dir = _resolve_run_dir(output_dir, config_path.stem, resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    copied_config_path = run_dir / "config.json"
    if config_path.resolve() != copied_config_path.resolve():
        shutil.copyfile(config_path, copied_config_path)

    seed = int(config.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = _resolve_device(config.get("device"))
    dtype_name, weight_dtype = _resolve_weight_dtype(config)
    model = _build_model(config).to(device=device, dtype=weight_dtype)
    optimizer = _build_optimizer(config, model)

    model_config = config.get("model", {})
    training_config = config.get("training", {})
    context_length = int(model_config.get("context_length", config.get("context_length")))
    batch_size = int(training_config.get("batch_size", config.get("batch_size", 32)))
    gradient_accumulation_steps = max(1, int(training_config.get("gradient_accumulation_steps", 1)))
    eval_iters = max(1, int(training_config.get("eval_iters", config.get("eval_iters", 10))))
    log_every = int(training_config.get("log_every", training_config.get("log_every_n_batches", 100)))
    performance_log_every = int(training_config.get("performance_log_every", 100))
    checkpoint_every = int(training_config.get("checkpoint_every", config.get("checkpoint_every", 0)))
    gradient_clip = training_config.get("gradient_clip", training_config.get("max_l2_norm"))

    train_path = _require_config_value(config, "train_dataset", "train_path", ("data", "train_path"))
    val_path = _require_config_value(config, "val_dataset", "val_path", ("data", "val_path"))
    train_data = _load_memmapped_npy(train_path, "training", context_length)
    val_data = _load_memmapped_npy(val_path, "validation", context_length)

    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = _compute_steps_per_epoch(config, train_data, effective_batch_size, context_length)
    max_iterations = _compute_max_iterations(config, steps_per_epoch)

    start_iteration = 0
    if resume is not None:
        resume_path = run_dir / "checkpoint_latest.pt" if str(resume) == "latest" else Path(resume)
        start_iteration = load_checkpoint(resume_path, model, optimizer)
        print(f"Resumed from {resume_path} at iteration {start_iteration}")

    metrics_path = run_dir / "metrics.csv"
    performance_path = run_dir / "performance.csv"
    plot_path = run_dir / "loss.svg"
    start_time = time.time()

    model.train()
    iteration = start_iteration
    start_epoch = start_iteration // steps_per_epoch
    epoch_slots = (max_iterations + steps_per_epoch - 1) // steps_per_epoch

    print(f"Training on {device} for up to {max_iterations} optimizer iterations")
    print(f"Weight dtype: {dtype_name}")
    print(f"Steps per epoch: {steps_per_epoch} optimizer iterations")
    print(
        f"Micro-batch size {batch_size}, gradient accumulation {gradient_accumulation_steps}, "
        f"effective batch size {effective_batch_size}"
    )
    min_learning_rate = float(training_config.get("min_learning_rate", optimizer.param_groups[0]["lr"]))
    warmup_iters = int(training_config.get("warmup_iters", 0))
    cosine_cycle_iters = int(training_config.get("cosine_cycle_iters", max_iterations))
    min_learning_rate_step = max(warmup_iters, cosine_cycle_iters)
    min_learning_rate_note = (
        f"reached after {min_learning_rate_step} optimizer steps"
        if min_learning_rate_step <= max_iterations
        else f"not reached within this run; scheduled after {min_learning_rate_step} optimizer steps"
    )
    print(
        f"Total optimizer steps: {max_iterations}; minimum learning rate {min_learning_rate:.3g} "
        f"{min_learning_rate_note}"
    )
    print(f"Writing outputs to {run_dir}")

    for epoch_index in range(start_epoch, epoch_slots):
        epoch_number = epoch_index + 1
        first_batch = start_iteration % steps_per_epoch if epoch_index == start_epoch else 0
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        for batch_index in range(first_batch, steps_per_epoch):
            if iteration >= max_iterations:
                break

            learning_rate = _learning_rate_for_iteration(config, iteration, max_iterations)
            _set_optimizer_lr(optimizer, learning_rate)

            next_iteration = iteration + 1
            should_log_performance = performance_log_every > 0 and next_iteration % performance_log_every == 0
            timings = {
                "data_wait": 0.0,
                "forward": 0.0,
                "backward": 0.0,
                "optimizer": 0.0,
            }
            if should_log_performance:
                _synchronize_device(device)
            step_start = time.perf_counter()
            optimizer.zero_grad()

            accumulated_loss = 0.0
            for _ in range(gradient_accumulation_steps):
                phase_start = time.perf_counter()
                inputs, targets = get_batch(train_data, batch_size, context_length, str(device))
                if should_log_performance:
                    _synchronize_device(device)
                timings["data_wait"] += time.perf_counter() - phase_start

                phase_start = time.perf_counter()
                logits = model(inputs)
                loss = _lm_loss(logits, targets)
                if should_log_performance:
                    _synchronize_device(device)
                timings["forward"] += time.perf_counter() - phase_start

                accumulated_loss += float(loss.item())
                phase_start = time.perf_counter()
                (loss / gradient_accumulation_steps).backward()
                if should_log_performance:
                    _synchronize_device(device)
                timings["backward"] += time.perf_counter() - phase_start

            if gradient_clip is not None:
                gradient_clipping(model.parameters(), float(gradient_clip))
            phase_start = time.perf_counter()
            optimizer.step()
            if should_log_performance:
                _synchronize_device(device)
            timings["optimizer"] = time.perf_counter() - phase_start
            timings["step"] = time.perf_counter() - step_start

            iteration += 1
            current_loss = accumulated_loss / gradient_accumulation_steps
            epoch_loss_sum += current_loss
            epoch_loss_count += 1
            total_processed_tokens = iteration * effective_batch_size * context_length

            if should_log_performance:
                _log_performance(
                    performance_path,
                    iteration,
                    epoch_number,
                    batch_index + 1,
                    batch_size,
                    gradient_accumulation_steps,
                    context_length,
                    timings,
                    device,
                )

            if log_every > 0 and iteration % log_every == 0:
                val_loss = _estimate_loss(model, val_data, batch_size, context_length, device, eval_iters)
                _log_metric(
                    metrics_path,
                    plot_path,
                    iteration,
                    epoch_number,
                    batch_index + 1,
                    "batch",
                    "train",
                    current_loss,
                    learning_rate,
                    total_processed_tokens,
                    start_time,
                )
                _log_metric(
                    metrics_path,
                    plot_path,
                    iteration,
                    epoch_number,
                    batch_index + 1,
                    "batch",
                    "val",
                    val_loss,
                    learning_rate,
                    total_processed_tokens,
                    start_time,
                )
                print(
                    f"iter {iteration}: train_loss={current_loss:.4f} "
                    f"val_loss={val_loss:.4f} lr={learning_rate:.3g} "
                    f"tokens={total_processed_tokens}"
                )

            if checkpoint_every > 0 and iteration % checkpoint_every == 0:
                _save_checkpoint_set(model, optimizer, iteration, run_dir)

        if epoch_loss_count > 0:
            learning_rate = _learning_rate_for_iteration(config, iteration, max_iterations)
            avg_train_loss = epoch_loss_sum / epoch_loss_count
            val_loss = _estimate_loss(model, val_data, batch_size, context_length, device, eval_iters)
            total_processed_tokens = iteration * effective_batch_size * context_length
            _log_metric(
                metrics_path,
                plot_path,
                iteration,
                epoch_number,
                steps_per_epoch,
                "epoch",
                "train",
                avg_train_loss,
                learning_rate,
                total_processed_tokens,
                start_time,
            )
            _log_metric(
                metrics_path,
                plot_path,
                iteration,
                epoch_number,
                steps_per_epoch,
                "epoch",
                "val",
                val_loss,
                learning_rate,
                total_processed_tokens,
                start_time,
            )
            _save_checkpoint_set(model, optimizer, iteration, run_dir, epoch=epoch_number)
            print(
                f"epoch {epoch_number}: train_loss={avg_train_loss:.4f} "
                f"val_loss={val_loss:.4f} tokens={total_processed_tokens}"
            )

        if iteration >= max_iterations:
            break

    save_checkpoint(model, optimizer, iteration, run_dir / "checkpoint_final.pt")
    save_checkpoint(model, optimizer, iteration, run_dir / "checkpoint_latest.pt")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TransformerLM from a JSON config.")
    parser.add_argument("configs", nargs="+", help="Path(s) to JSON training config files.")
    parser.add_argument(
        "--batch_mode",
        "--batch-mode",
        action="store_true",
        help="Run multiple config files sequentially in one command.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume from a checkpoint path. If no path is supplied, uses checkpoint_latest.pt in the run folder.",
    )
    args = parser.parse_args()
    if len(args.configs) > 1 and not args.batch_mode:
        parser.error("multiple config paths require --batch_mode")
    return args


def main() -> None:
    args = parse_args()
    if args.batch_mode:
        print(f"Batch mode: running {len(args.configs)} configs sequentially")
        run_dirs = []
        for index, config_path in enumerate(args.configs, start=1):
            print(f"\nBatch run {index}/{len(args.configs)}: {config_path}")
            run_dir = train(config_path, resume=args.resume)
            run_dirs.append(run_dir)
            print(f"Batch run {index}/{len(args.configs)} done. Run artifacts are in {run_dir}")

        print("Batch mode complete. Run artifacts:")
        for run_dir in run_dirs:
            print(f"  {run_dir}")
        return

    run_dir = train(args.configs[0], resume=args.resume)
    print(f"Done. Run artifacts are in {run_dir}")


if __name__ == "__main__":
    main()
