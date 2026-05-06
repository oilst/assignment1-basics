import torch
import torch.nn.functional as F
from einops import einsum
from jaxtyping._array_types import Float, Int
from torch import Tensor


def cross_entropy(inputs: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]):
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
    x_max = torch.max(inputs, dim=1, keepdim=True).values
    diff = inputs - x_max
    x_exp = torch.exp(diff)
    #nice trick by abhinav to slice the GT value
    return -(diff[torch.arange(inputs.shape[0]), targets] - torch.log(x_exp.sum(dim=1))).mean()
