import torch
import torch.nn.functional as F
from einops import einsum


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute mean cross-entropy for logits of shape ``(..., vocab_size)``.

    Args:
        logits: Unnormalized class scores with vocabulary on the last dimension.
            Shape: (..., vocab_size)
        targets: Correct class index for each example in the leading batch dimensions.
            Shape: (...)

    Returns:
        Scalar tensor containing the average negative log-likelihood over all examples.
    """
    # Loss - log (exp(l_k) / Sum_j exp(l_j)) = log (Sum_j exp(l_j)) - l_k if k is the index of the gt
    shifted_logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    target_mask = F.one_hot(targets, num_classes=shifted_logits.size(-1)).to(shifted_logits.dtype)
    target_logits = einsum(shifted_logits, target_mask, "... vocab_size,... vocab_size->...")
    log_normalizer = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))

    return (log_normalizer - target_logits).mean()
