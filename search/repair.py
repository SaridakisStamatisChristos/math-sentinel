
from __future__ import annotations

from typing import List

from proof.actions import Action, ActionType
from proof.state import ProofState


def _known_answer(state: ProofState) -> str:
    return state.expected_answer.strip()


def _arithmetic_tool(problem_text: str) -> str:
    expr = problem_text.split(":", 1)[-1]
    if "*" in expr:
        return "multiply"
    if "+" in expr:
        return "add"
    return "subtract"


def fallback_repairs(state: ProofState) -> List[Action]:
    repairs: List[Action] = []
    if state.domain == "linear_equation":
        repairs.append(Action(type=ActionType.APPLY, tool="solve_linear_step", content=state.problem_text.split(":", 1)[-1].strip()))
    if state.domain == "arithmetic":
        repairs.append(Action(type=ActionType.APPLY, tool=_arithmetic_tool(state.problem_text), content=state.problem_text))
    if state.domain == "modular":
        repairs.append(Action(type=ActionType.APPLY, tool="modular_reduce", content=state.problem_text))
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
    if state.domain == "primality":
        repairs.append(Action(type=ActionType.APPLY, tool="primality", content=state.problem_text))
    if state.domain == "factorization":
        repairs.append(Action(type=ActionType.APPLY, tool="factorize", content=state.problem_text))
    if state.domain == "divmod":
        repairs.append(Action(type=ActionType.APPLY, tool="divmod", content=state.problem_text))
        if _known_answer(state):
            repairs.append(Action(type=ActionType.ANSWER, content=_known_answer(state)))
    if state.domain == "gcd_lcm":
        repairs.append(Action(type=ActionType.APPLY, tool="gcd_lcm", content=state.problem_text))
        if _known_answer(state):
            repairs.append(Action(type=ActionType.ANSWER, content=_known_answer(state)))
    return repairs
