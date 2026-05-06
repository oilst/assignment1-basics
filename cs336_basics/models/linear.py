import math

import torch
from torch import nn, einsum


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device=None, dtype: torch.dtype=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        std = math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #use einops
        return einsum('... i, o i -> ... o', x, self.weight)