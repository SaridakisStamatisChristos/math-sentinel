
from __future__ import annotations

from typing import List

from .state import ProofState


def is_solved(state: ProofState) -> bool:
    return state.status == "solved"


def add_subgoal(state: ProofState, subgoal: str) -> None:
    if subgoal and subgoal not in state.subgoals:
        state.subgoals.append(subgoal)


def resolve_subgoal(state: ProofState, subgoal: str) -> None:
    state.subgoals = [x for x in state.subgoals if x != subgoal]


def estimate_goal_complexity(state: ProofState) -> float:
    complexity = 0.0
    complexity += len(state.problem_text) / 60.0
    complexity += len(state.subgoals) * 0.5
    complexity += len(state.derived_facts) * 0.15
    return complexity
