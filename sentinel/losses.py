
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_ce(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    bsz, timesteps, vocab = logits.shape
    return F.cross_entropy(logits.reshape(bsz * timesteps, vocab), targets.reshape(bsz * timesteps), ignore_index=pad_id)


def verifier_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


def verifier_focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = targets * probs + (1.0 - targets) * (1.0 - probs)
    alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    loss = alpha_t * ((1.0 - pt) ** gamma) * ce
    return loss.mean()


def ranking_margin_loss(good_scores: torch.Tensor, bad_scores: torch.Tensor, margin: float = 0.25) -> torch.Tensor:
    return torch.relu(margin - (good_scores - bad_scores)).mean()


def verifier_pairwise_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    pos_targets: torch.Tensor,
    neg_targets: torch.Tensor,
    *,
    margin: float = 0.6,
    rank_weight: float = 0.35,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = verifier_focal_bce_loss(pos_logits, pos_targets, gamma=focal_gamma, alpha=focal_alpha)
    bce = bce + verifier_focal_bce_loss(neg_logits, neg_targets, gamma=focal_gamma, alpha=focal_alpha)

    pos_probs = torch.sigmoid(pos_logits)
    neg_probs = torch.sigmoid(neg_logits)

    pos_quality = pos_logits[:, 0] + pos_logits[:, 1] + pos_logits[:, 2] + 0.5 * pos_logits[:, 4] - pos_logits[:, 3]
    neg_quality = neg_logits[:, 0] + neg_logits[:, 1] + neg_logits[:, 2] + 0.5 * neg_logits[:, 4] - neg_logits[:, 3]

    rank = ranking_margin_loss(pos_quality, neg_quality, margin=margin)
    rank = rank + 0.5 * ranking_margin_loss(pos_logits[:, 0], neg_logits[:, 0], margin=margin * 0.5)
    rank = rank + 0.5 * ranking_margin_loss(pos_logits[:, 2], neg_logits[:, 2], margin=margin * 0.5)
    rank = rank + 0.5 * ranking_margin_loss(-pos_logits[:, 3], -neg_logits[:, 3], margin=margin * 0.5)

    total = bce + rank_weight * rank
    return total, bce.detach(), rank.detach()
