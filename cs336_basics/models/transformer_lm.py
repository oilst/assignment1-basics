from torch import nn

from cs336_basics.models.embedding import Embedding
from cs336_basics.models.linear import Linear
from cs336_basics.models.rms_norm import RMSNorm
from cs336_basics.models.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float):
        super().__init__()
        self.rope_theta = rope_theta
        self.token_embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids):
        x = self.token_embedding(token_ids)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x)
