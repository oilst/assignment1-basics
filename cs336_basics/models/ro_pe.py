import torch
import torch.nn as nn
from einops import rearrange, einsum

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        pos = torch.arange(max_seq_len, device=device).float()          # (T,)
        dim = torch.arange(d_k // 2, device=device).float()             # (D/2,)
        inv_freq = theta ** (-2 * dim / d_k)                            # (D/2,)

        angles = einsum(pos, inv_freq, 't, d -> t d')                   # (T, D/2)

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_k)
        token_positions: (...) matching x.shape[:-1]
        """
        assert x.shape[-1] == self.d_k

        cos = self.cos[token_positions]   # (..., d_k/2)
        sin = self.sin[token_positions]   # (..., d_k/2)

        x_pair = rearrange(x, '... (d pair) -> ... d pair', pair=2)
        x1, x2 = x_pair[..., 0], x_pair[..., 1]

        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin

        y = torch.stack((y1, y2), dim=-1)
        return rearrange(y, '... d pair -> ... (d pair)')