import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import BinaryIO
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_file(
    file_path: str | os.PathLike,
    num_processes: int = 20,
    split_special_token: bytes = b"<|endoftext|>",
) -> dict[tuple[int, ...], int]:
    """
    Returns:
        A dictionary mapping each pre-token (represented as a tuple of byte values)
        to the number of times it occurs.
    """
    if num_processes <= 0:
        raise ValueError("num_processes must be a positive integer")

    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token)

    chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
    if not chunk_ranges:
        return {}

    # Each chunk range is handled in its own worker process.
    worker_inputs = [
        (str(file_path), start, end, split_special_token)
        for start, end in chunk_ranges
    ]
    max_workers = min(len(worker_inputs), os.cpu_count() or 1)
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            chunk_counts = list(executor.map(_pretokenize_chunk, worker_inputs))
    except (PermissionError, RuntimeError, BrokenProcessPool):
        # Fallback for restricted or non-safe multiprocessing entrypoints.
        chunk_counts = [_pretokenize_chunk(item) for item in worker_inputs]

    token_counts: dict[tuple[int, ...], int] = defaultdict(int)
    for chunk_count in chunk_counts:
        for token, count in chunk_count.items():
            token_counts[token] += count

    return dict(token_counts)


def _pretokenize_chunk(
    args: tuple[str, int, int, bytes],
) -> dict[tuple[int, ...], int]:
    file_path, start, end, split_special_token = args
    split_special_token_str = split_special_token.decode("utf-8")

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    token_counts: dict[tuple[int, ...], int] = defaultdict(int)
    parts = chunk.split(split_special_token_str)
    for i, part in enumerate(parts):
        for token in re.findall(PAT, part):
            token_counts[tuple(token.encode("utf-8"))] += 1
        if i < len(parts) - 1:
            token_counts[tuple(split_special_token)] += 1

    return dict(token_counts)
