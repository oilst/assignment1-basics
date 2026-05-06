import torch
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
        self.vocab_size = vocab_size
        self.context_length = context_length
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

    @staticmethod
    def _apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        keep_sorted = (cumulative_probs - sorted_probs) < top_p
        keep_sorted[..., 0] = True

        filtered_sorted_probs = sorted_probs.masked_fill(~keep_sorted, 0.0)
        filtered_probs = torch.zeros_like(probs)
        filtered_probs.scatter_(-1, sorted_indices, filtered_sorted_probs)
        return filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def decode(
        self,
        prompt: torch.Tensor | list[int] | list[list[int]],
        max_new_tokens: int,
        eos_token_id: int | None = None,
        temperature: float = 1.0,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Generate token IDs from a prompt using temperature and optional top-p sampling.

        Args:
            prompt: Token IDs with shape ``(sequence_length,)`` or
                ``(batch_size, sequence_length)``.
            max_new_tokens: Maximum number of tokens to append.
            eos_token_id: If provided, stop once every sequence has generated
                this token.
            temperature: Softmax temperature. ``0`` uses greedy argmax.
            top_p: If provided, nucleus sampling threshold in ``(0, 1]``.

        Returns:
            A tensor containing the prompt followed by generated token IDs.
            A 1D prompt returns a 1D tensor; a batched prompt returns a 2D tensor.
        """
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

        device = next(self.parameters()).device
        generated = torch.as_tensor(prompt, dtype=torch.long, device=device)
        is_batched = generated.ndim == 2
        if generated.ndim == 1:
            generated = generated.unsqueeze(0)
        elif generated.ndim != 2:
            raise ValueError("prompt must be a 1D or 2D sequence of token IDs")
        if generated.shape[1] == 0:
            raise ValueError("prompt must contain at least one token")

        finished = torch.zeros(generated.shape[0], dtype=torch.bool, device=device)
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                model_input = generated[:, -self.context_length :]
                next_token_logits = self(model_input)[:, -1, :]

                if temperature == 0:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    if top_p is not None and top_p < 1:
                        probs = self._apply_top_p(probs, top_p)
                    next_token = torch.multinomial(probs, num_samples=1)

                if eos_token_id is not None:
                    eos_fill = torch.full_like(next_token, eos_token_id)
                    next_token = torch.where(finished.unsqueeze(-1), eos_fill, next_token)

                generated = torch.cat([generated, next_token], dim=1)

                if eos_token_id is not None:
                    finished |= next_token.squeeze(-1) == eos_token_id
                    if torch.all(finished):
                        break
        finally:
            if was_training:
                self.train()

        if is_batched:
            return generated
        return generated.squeeze(0)
