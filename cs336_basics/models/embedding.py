import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        self.weights = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.weights, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, T]
        # output: [B, T, D]
        return self.weights[token_ids]