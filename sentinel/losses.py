
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


def verifier_pairwise_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    pos_targets: torch.Tensor,
    neg_targets: torch.Tensor,
    *,
    margin: float = 0.2,
    rank_weight: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = verifier_bce_loss(pos_logits, pos_targets) + verifier_bce_loss(neg_logits, neg_targets)

    pos_probs = torch.sigmoid(pos_logits)
    neg_probs = torch.sigmoid(neg_logits)

    pos_quality = pos_probs[:, 0] + pos_probs[:, 1] + pos_probs[:, 2] + 0.5 * pos_probs[:, 4] - pos_probs[:, 3]
    neg_quality = neg_probs[:, 0] + neg_probs[:, 1] + neg_probs[:, 2] + 0.5 * neg_probs[:, 4] - neg_probs[:, 3]

    rank = ranking_margin_loss(pos_quality, neg_quality, margin=margin)
    rank = rank + 0.5 * ranking_margin_loss(pos_probs[:, 0], neg_probs[:, 0], margin=margin * 0.5)
    rank = rank + 0.5 * ranking_margin_loss(pos_probs[:, 2], neg_probs[:, 2], margin=margin * 0.5)
    rank = rank + 0.5 * ranking_margin_loss(1.0 - pos_probs[:, 3], 1.0 - neg_probs[:, 3], margin=margin * 0.5)

    total = bce + rank_weight * rank
    return total, bce.detach(), rank.detach()
