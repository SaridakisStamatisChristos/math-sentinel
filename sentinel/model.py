
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).view(1, 1, seq_len, seq_len)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, timesteps, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, timesteps, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, timesteps, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, timesteps, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.mask[:, :, :timesteps, :timesteps], self._mask_fill_value(att))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, timesteps, channels)
        return self.proj(y)

    @staticmethod
    def _mask_fill_value(att: torch.Tensor) -> float:
        return torch.finfo(att.dtype).min


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, seq_len, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        _, timesteps = x.shape
        pos = torch.arange(timesteps, device=x.device).unsqueeze(0)
        h = self.drop(self.tok(x) + self.pos(pos))
        for blk in self.blocks:
            h = blk(h)
        return self.ln(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_hidden(x))

    @torch.no_grad()
    def generate_ids(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 24,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        x = input_ids
        for _ in range(max_new_tokens):
            if x.shape[1] > self.seq_len:
                x = x[:, -self.seq_len :]
            logits = self(x)[:, -1, :]
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                    thresh = values[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < thresh, torch.full_like(logits, -float("inf")), logits)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_id], dim=1)
            if eos_id is not None and int(next_id.item()) == eos_id:
                break
        return x
