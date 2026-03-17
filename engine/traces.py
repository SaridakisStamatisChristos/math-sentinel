from __future__ import annotations

from typing import List

from .state import ReasoningState


def render_human_trace(state: ReasoningState) -> str:
    lines: List[str] = []
    lines.append("=== PROBLEM ===")
    lines.append(state.problem_text)
    lines.append("=== GOAL ===")
    lines.append(state.goal)
    lines.append("=== ACTIONS ===")
    for idx, rec in enumerate(state.action_history):
        lines.append(f"{idx+1}. {rec}")
    lines.append("=== TOOLS ===")
    for idx, rec in enumerate(state.tool_history):
        lines.append(f"{idx+1}. {rec}")
    lines.append("=== FINAL ===")
    lines.append(f"status={state.status} answer={state.final_answer}")
    return "\n".join(lines)


def render_machine_trace(state: ReasoningState) -> str:
    return state.serialize()
