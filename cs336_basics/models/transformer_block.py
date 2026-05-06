import torch
from jaxtyping._array_types import Float
from torch import nn, Tensor

from cs336_basics.models.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.models.rms_norm import RMSNorm
from cs336_basics.models.swi_glu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, seq_len, theta):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ln_1 = RMSNorm(d_model)
        self.ln_2 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len=seq_len, theta=theta)
        self.swi_glu = SwiGLU(d_model, d_ff)

    def forward(self, x: Float[Tensor, " batch sequence_length d_model"]) -> Float[Tensor, " batch sequence_length d_model"]:
        attn_out = self.attn(self.ln_1(x))  # (B, T, d_model)
        out1 = x + attn_out  # (B, T, d_model)
        out2 = self.swi_glu(self.ln_2(out1))
        return out1 + out2


