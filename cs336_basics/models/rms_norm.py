import torch
from einops import reduce
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device = None, dtype: torch.dtype = None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()  # compute in fp32 for stability

        rms = torch.sqrt(
            reduce(x ** 2, 'b t d -> b t 1', 'mean') + self.eps
        )
        out = x / rms * self.weight

        return out.to(in_dtype)