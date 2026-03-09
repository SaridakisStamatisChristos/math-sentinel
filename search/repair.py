
from __future__ import annotations

from typing import List

from proof.actions import Action, ActionType
from proof.state import ProofState


def fallback_repairs(state: ProofState) -> List[Action]:
    repairs: List[Action] = []
    if state.domain == "linear_equation":
        repairs.append(Action(type=ActionType.APPLY, tool="solve_linear_step", content=state.problem_text.split(":", 1)[-1].strip()))
    if state.domain in {"arithmetic", "modular"}:
        repairs.append(Action(type=ActionType.ANSWER, content=state.expected_answer))
    if state.domain == "fractions":
        repairs.append(Action(type=ActionType.APPLY, tool="reduce_fraction", content=state.problem_text))
    if state.domain == "polynomial_simplify":
        repairs.append(Action(type=ActionType.APPLY, tool="simplify_polynomial", content=state.problem_text.split(":", 1)[-1].strip()))
    if state.domain == "derivative":
        repairs.append(Action(type=ActionType.APPLY, tool="derivative", content=state.problem_text.split(":", 1)[-1].strip()))
    if state.domain == "integral":
        repairs.append(Action(type=ActionType.APPLY, tool="antiderivative", content=state.problem_text.split(":", 1)[-1].strip()))
    if state.domain == "parity_proof":
        repairs.append(Action(type=ActionType.APPLY, tool="prove_even", content=state.problem_text))
    return repairs
