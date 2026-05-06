import math

import torch
from torch import nn
from einops import einsum
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, Mask: torch.Tensor) -> torch.Tensor:
    d_k = Q.size(-1)


    scores = einsum(Q, K, 'b ... n d_k, b ... m d_k -> b ... n m') / math.sqrt(d_k)
    scores = scores.masked_fill(~Mask, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    output = einsum(attn_weights, V, 'b ... n m, b ... m d_v -> b ... n d_v')
    return output