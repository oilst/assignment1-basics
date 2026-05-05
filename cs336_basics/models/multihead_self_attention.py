import torch
from torch import nn

from cs336_basics.models.linear import Linear
from cs336_basics.models.ro_pe import RoPE
from cs336_basics.models.scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len, theta: float = None, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // num_heads
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones((max_seq_len, max_seq_len), device=device, dtype=torch.bool)),
            persistent=False,
        )
        self.W_K = Linear(d_model, num_heads*self.head_dim)
        self.W_Q = Linear(d_model, num_heads*self.head_dim)
        self.W_V = Linear(d_model, num_heads*self.head_dim)
        self.W_O = Linear(num_heads*self.head_dim, d_model)
        self.ro_pe = None
        if theta is not None:
            self.ro_pe = RoPE(theta, self.head_dim, max_seq_len, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if T > self.causal_mask.shape[0] and self.ro_pe is not None:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")
        if T > self.causal_mask.shape[0]:
            self.causal_mask = torch.tril(torch.ones((T, T), device=x.device, dtype=torch.bool))
        K = self.W_K(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        Q = self.W_Q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        V = self.W_V(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)

        # add rope
        if self.ro_pe is not None:
            token_positions = torch.arange(T, device=x.device)  # (T,)
            K = self.ro_pe(K, token_positions)  # (B, num_heads, T, head_dim)
            Q = self.ro_pe(Q, token_positions)  # (B, num_heads, T, head_dim)

        # k should have dimensions (B, num_heads, T, head_dim)
        # q should have dimensions (B, num_heads, T, head_dim)
        # attn should have dimensions (B, num_heads, T, T)
        causal_mask = self.causal_mask[:T, :T]
        attn = scaled_dot_product_attention(Q, K, V, causal_mask)  # (B, num_heads, T, head_dim)
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B, T, d_model)

        return self.W_O(attn)
