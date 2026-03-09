
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class StateVerifier(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 160, dropout: float = 0.1) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.emb(x)
        h, _ = self.rnn(emb)
        pooled = h.mean(dim=1)
        return self.head(self.drop(pooled))

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
