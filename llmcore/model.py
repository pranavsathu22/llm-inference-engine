"""Days 2–5 — the model, built bottom-up.

Head (day 2) -> MultiHeadAttention (day 3) -> Block (day 4) -> GPT (day 5).

Shape convention throughout: B = batch, T = time/sequence, C = n_embd (channels).
Write the shape of every tensor in a comment as you go — shape bugs are 90% of the
pain here, and naming the shapes is how you catch them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128      # max context length
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.0


class Head(nn.Module):
    """Day 2 — one head of self-attention.

    Linear projections to key/query/value, scaled dot-product attention over
    (B, T, head_size), a causal mask so position t only sees <= t, softmax, then a
    weighted sum of values. Register the causal mask as a buffer (it's not a parameter).
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.head_size = head_size
        
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(
        self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x: (B, T, n_embd) -> out: (B, T, head_size), cache: (k, v) each (B, T_total, head_size)
        # scores = q @ k.transpose(-2,-1) / sqrt(head_size); mask future; softmax; @ v
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        if past_kv:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        # allows for inner two dimensions to match and to compute dot product
        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.head_size ** 0.5)

        #multiply by mask to hide fture tokens
        T_new = q.shape[1]
        T_total = k.shape[1]

        T_cached = k.shape[1] - q.shape[1]

        causal_mask = self.mask[T_cached : T_cached + T_new, :T_total]
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=2)
        out = attn @ v

        return out, (k, v)

class MultiHeadAttention(nn.Module):
    """Day 3 — n_head heads in parallel, concatenated then projected back to n_embd."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, n_embd // n_head, block_size, dropout) for i in range(n_head)])
        self.out_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor, past_kv: list | None = None) -> tuple[torch.Tensor, list]:
        # concat head outputs on the channel dim, then project. (B,T,C) -> (B,T,C)
        outputs = []
        new_caches = []

        for i, head in enumerate(self.heads):
            head_past = past_kv[i] if past_kv is not None else None
            out, cache = head(x, past_kv=head_past)
            outputs.append(out)
            new_caches.append(cache)

        out = torch.cat(outputs, dim=2)
        return self.out_proj(out), new_caches

class FeedForward(nn.Module):
    """Day 4 — position-wise MLP: Linear -> GELU/ReLU -> Linear, usually 4x inner width."""

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.sequential(x)
        return out

class Block(nn.Module):
    """Day 4 — a transformer block with PRE-norm residuals:
        x = x + attn(ln1(x))
        x = x + ffwd(ln2(x))
    Pre-norm (LN before the sublayer) trains far more stably than post-norm — try both
    and watch the loss if you want to feel why.
    """
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.gptConfig = cfg
        self.ff = FeedForward(self.gptConfig.n_embd, self.gptConfig.dropout)
        self.attn = MultiHeadAttention(self.gptConfig.n_embd, self.gptConfig.n_head, self.gptConfig.block_size, self.gptConfig.dropout)
        self.ln1 = nn.LayerNorm(self.gptConfig.n_embd)
        self.ln2 = nn.LayerNorm(self.gptConfig.n_embd)

    def forward(self, x: torch.Tensor, past_kv: list | None = None) -> tuple[torch.Tensor, list]:
        attn_out, new_cache = self.attn(self.ln1(x), past_kv=past_kv)

        x = x + attn_out
        x = x + self.ff(self.ln2(x))

        return x, new_cache

class GPT(nn.Module):
    """Day 5 — the whole model.

    token embedding + positional embedding -> n_layer Blocks -> final LayerNorm ->
    LM head (Linear n_embd -> vocab_size). Tie the LM head weight to the token
    embedding. forward returns logits, and loss if targets are given.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.t_embd = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_embd = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.stack = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.fln = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        self.lm_head.weight = self.t_embd.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        # PyTorch's default inits (std=1 for Embedding) are far too large once the
        # embedding table is reused as the LM head weight via tying -- rescale
        # everything to GPT-2/nanoGPT's std=0.02 so logits start well-behaved.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, past_kv=None
    ) -> tuple[torch.Tensor, torch.Tensor | None, list]:
        # idx: (B, T) -> logits: (B, T, vocab_size); loss: scalar cross-entropy or None
        B, T = idx.shape

        if past_kv is None:
            offset = 0
        else:
            offset = past_kv[0][0][0].shape[1]

        tok = self.t_embd(idx)
        positions = torch.arange(offset, offset + T, device=idx.device)
        pos = self.pos_embd(positions)
        x = tok + pos

        #run thru blocks and final layernorm
        new_caches = []
        for i, block in enumerate(self.stack):
            block_past = past_kv[i] if past_kv is not None else None
            x, new_cache = block(x, past_kv=block_past)
            new_caches.append(new_cache)

        x = self.fln(x)
        logits = self.lm_head(x)

        loss = None

        #calculate loss
        if targets is not None:
            flattened_logits = logits.view(B*T, self.cfg.vocab_size)
            flattened_targets = targets.view(B*T)
            loss = F.cross_entropy(flattened_logits, flattened_targets)

        return logits, loss, new_caches
