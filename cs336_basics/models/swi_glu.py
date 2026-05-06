import torch
from torch import nn

from cs336_basics.models.linear import Linear


class SwiGLU(nn.Module):
    def __init__(self, in_features: int, d_ff: int = None, device: torch.device = None, dtype: torch.dtype = None):
        super(SwiGLU, self).__init__()
        if d_ff is None:
            d_ff = 8/3 * in_features
            # round up to multple of 64
            d_ff = int((d_ff + 63) // 64 * 64)
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1 = Linear(in_features=in_features, out_features=self.d_ff)
        self.w3 = Linear(in_features=in_features, out_features=self.d_ff)
        self.w2 = Linear(in_features=self.d_ff, out_features=in_features)


    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))

