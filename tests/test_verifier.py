from __future__ import annotations

import torch

from sentinel.verifier import StateVerifier


def test_state_verifier_forward_and_scores_shape() -> None:
    verifier = StateVerifier(vocab_size=32, hidden_size=48, n_heads=4, n_layers=2, max_seq_len=64)
    x = torch.randint(0, 32, (3, 10))

    logits = verifier(x)
    scores = verifier.predict_scores(x)

    assert logits.shape == (3, 5)
    assert all(k in scores for k in ["valid_step_prob", "goal_progress_score", "proof_completion_score", "risk_score", "branch_priority"])
    assert scores["valid_step_prob"].shape == (3,)
