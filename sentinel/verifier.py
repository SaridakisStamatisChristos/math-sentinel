
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class StateVerifier(nn.Module):
    """
    A stronger, transformer-style verifier with attention pooling.

    - Token + positional embeddings
    - Norm-first Transformer encoder layers (bidirectional)
    - Learnable attention pooling query for global summary
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 160,
        dropout: float = 0.1,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 256,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok = nn.Embedding(vocab_size, hidden_size)
        self.pos = nn.Embedding(max_seq_len, hidden_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=ff_mult * hidden_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.pool_attn = nn.MultiheadAttention(hidden_size, n_heads, dropout=dropout, batch_first=True)

        self.drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_size * 2, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, timesteps = x.shape
        pos_ids = torch.arange(timesteps, device=x.device)
        pos_ids = torch.clamp(pos_ids, max=self.max_seq_len - 1)
        h = self.tok(x) + self.pos(pos_ids).unsqueeze(0)

        cls = self.cls.expand(bsz, 1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.encoder(self.drop(h))

        q = self.pool_query.expand(bsz, 1, -1)
        pooled, _ = self.pool_attn(q, h, h)
        pooled = pooled.squeeze(1)

        feats = self.mlp(self.drop(pooled))
        return self.head(feats)

    def predict_scores(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self(x)
        probs = torch.sigmoid(logits)
        return {
            "valid_step_prob": probs[:, 0],
            "goal_progress_score": probs[:, 1],
            "proof_completion_score": probs[:, 2],
            "risk_score": probs[:, 3],
            "branch_priority": probs[:, 4],
        }
