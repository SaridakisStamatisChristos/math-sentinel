from __future__ import annotations

import torch

from sentinel.losses import verifier_pairwise_loss


def test_verifier_pairwise_loss_prefers_separated_logits() -> None:
    pos_targets = torch.tensor([[0.98, 0.95, 0.98, 0.05, 0.9]], dtype=torch.float32)
    neg_targets = torch.tensor([[0.08, 0.12, 0.02, 0.92, 0.08]], dtype=torch.float32)

    good_total, _, _ = verifier_pairwise_loss(
        torch.tensor([[3.0, 2.5, 3.0, -2.5, 2.0]], dtype=torch.float32),
        torch.tensor([[-2.5, -2.0, -3.0, 2.5, -2.0]], dtype=torch.float32),
        pos_targets,
        neg_targets,
    )
    bad_total, _, _ = verifier_pairwise_loss(
        torch.zeros((1, 5), dtype=torch.float32),
        torch.zeros((1, 5), dtype=torch.float32),
        pos_targets,
        neg_targets,
    )

    assert good_total.item() < bad_total.item()
