
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_ce(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    bsz, timesteps, vocab = logits.shape
    return F.cross_entropy(logits.reshape(bsz * timesteps, vocab), targets.reshape(bsz * timesteps), ignore_index=pad_id)


def verifier_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


def ranking_margin_loss(good_scores: torch.Tensor, bad_scores: torch.Tensor, margin: float = 0.25) -> torch.Tensor:
    return torch.relu(margin - (good_scores - bad_scores)).mean()
