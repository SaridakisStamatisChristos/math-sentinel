
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
    value_weight: float = 0.35,
    novelty_weight: float = 0.08,
    obligation_weight: float = 0.08,
    evidence_weight: float = 0.10,
    stagnation_penalty: float = 0.18,
    repeat_penalty: float = 0.10,
    depth: int = 1,
    solved: bool = False,
) -> float:
    valid = float(verifier_scores.get("valid_step_prob", 0.5))
    progress = float(verifier_scores.get("goal_progress_score", 0.5))
    completion = float(verifier_scores.get("proof_completion_score", 0.5))
    priority = float(verifier_scores.get("branch_priority", 0.5))
    value_estimate = float(verifier_scores.get("value_estimate", priority))
    risk = float(verifier_scores.get("risk_score", 0.5))
    exact_progress = float(exec_info.get("goal_progress", 0.0))
    local_valid = float(exec_info.get("valid_step", 1.0))
    answer_present = float(exec_info.get("answer_present", 0.0))
    tactic_bias = float(exec_info.get("tactic_bias", 0.5))
    novelty_bonus = float(exec_info.get("novelty_bonus", 0.5))
    tool_bias = float(exec_info.get("tool_bias", 0.5))
    evidence_bonus = float(exec_info.get("evidence_bonus", 0.0))
    obligation_progress = float(exec_info.get("obligation_progress", 0.0))
    stagnation = float(exec_info.get("stagnation_penalty", 0.0))
    repeat_bias = float(exec_info.get("repeat_penalty", 0.0))

    score = 0.0
    score += 0.45 * valid
    score += goal_bonus * progress
    score += 0.20 * completion
    score += 0.15 * priority
    score += value_weight * value_estimate
    score += 0.30 * exact_progress
    score += completion_bonus * answer_present
    score += tactic_bonus * (tactic_bias - 0.5)
    score += novelty_weight * (novelty_bonus - 0.5)
    score += 0.08 * (tool_bias - 0.5)
    score += evidence_weight * evidence_bonus
    score += obligation_weight * obligation_progress
    score -= 0.25 * risk
    score -= stagnation_penalty * stagnation
    score -= repeat_penalty * repeat_bias
    score -= simplicity_penalty * depth
    if local_valid < 0.5:
        score -= invalid_penalty
    if not solved and answer_present < 0.5 and exact_progress > 0.0 and depth > 1:
        score -= incomplete_penalty * min(depth - 1, 2)
    if solved:
        score += solved_bonus
    return float(score)
