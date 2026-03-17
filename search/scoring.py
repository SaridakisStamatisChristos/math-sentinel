
from __future__ import annotations

from typing import Dict


def combine_scores(
    verifier_scores: Dict[str, float],
    exec_info: Dict[str, float],
    simplicity_penalty: float = 0.02,
    invalid_penalty: float = 1.0,
    goal_bonus: float = 0.4,
    solved_bonus: float = 1.0,
    completion_bonus: float = 0.15,
    incomplete_penalty: float = 0.35,
    tactic_bonus: float = 0.12,
    depth: int = 1,
    solved: bool = False,
) -> float:
    valid = float(verifier_scores.get("valid_step_prob", 0.5))
    progress = float(verifier_scores.get("goal_progress_score", 0.5))
    completion = float(verifier_scores.get("proof_completion_score", 0.5))
    priority = float(verifier_scores.get("branch_priority", 0.5))
    risk = float(verifier_scores.get("risk_score", 0.5))
    exact_progress = float(exec_info.get("goal_progress", 0.0))
    local_valid = float(exec_info.get("valid_step", 1.0))
    answer_present = float(exec_info.get("answer_present", 0.0))
    tactic_bias = float(exec_info.get("tactic_bias", 0.5))

    score = 0.0
    score += 0.45 * valid
    score += goal_bonus * progress
    score += 0.20 * completion
    score += 0.15 * priority
    score += 0.30 * exact_progress
    score += completion_bonus * answer_present
    score += tactic_bonus * (tactic_bias - 0.5)
    score -= 0.25 * risk
    score -= simplicity_penalty * depth
    if local_valid < 0.5:
        score -= invalid_penalty
    if not solved and answer_present < 0.5 and exact_progress > 0.0 and depth > 1:
        score -= incomplete_penalty * min(depth - 1, 2)
    if solved:
        score += solved_bonus
    return float(score)
