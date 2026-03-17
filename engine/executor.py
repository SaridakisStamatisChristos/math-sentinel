from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from .actions import Action, ActionType
from .goals import add_subgoal, resolve_subgoal
from .state import ReasoningState


AnswerJudge = Callable[[ReasoningState, str], bool]


def default_answer_judge(state: ReasoningState, candidate: str) -> bool:
    return candidate.strip() == state.expected_answer.strip()


class StateExecutor:
    def __init__(self, tool_registry: Any, answer_judge: Optional[AnswerJudge] = None) -> None:
        self.tool_registry = tool_registry
        self.answer_judge = answer_judge or default_answer_judge

    def apply(self, state: ReasoningState, action: Action) -> Tuple[ReasoningState, Dict[str, Any]]:
        child = state.clone()
        child.action_history.append(action.to_record())
        info: Dict[str, Any] = {
            "valid_step": 1.0,
            "goal_progress": 0.0,
            "proof_completion": 1.0 if child.status == "solved" else 0.0,
            "risk": 0.0,
            "note": "",
        }

        try:
            if action.type == ActionType.THINK:
                info["note"] = action.content[:200]
            elif action.type == ActionType.SUBGOAL:
                add_subgoal(child, action.content)
                info["goal_progress"] = 0.1
            elif action.type == ActionType.RESOLVE_SUBGOAL:
                resolve_subgoal(child, action.content)
                info["goal_progress"] = 0.2
            elif action.type == ActionType.LEMMA:
                child.lemma_refs.append(action.content.strip() or action.name or "anonymous_lemma")
                info["goal_progress"] = 0.05
            elif action.type == ActionType.ASSUME:
                child.assumptions.append(action.content)
                info["goal_progress"] = 0.02
            elif action.type in {ActionType.APPLY, ActionType.CALL_PLUGIN, ActionType.CHECK, ActionType.REWRITE, ActionType.SIMPLIFY}:
                result = self.tool_registry.call(action.tool, action.content, child)
                child.tool_history.append({"tool": action.tool, "input": action.content, "result": result})
                if result.get("ok"):
                    rendered = result.get("result", "")
                    if rendered:
                        child.derived_facts.append(str(rendered))
                    info["goal_progress"] = float(result.get("goal_progress", 0.2))
                    if result.get("solved"):
                        child.final_answer = str(result.get("answer", rendered))
                        child.status = "solved"
                else:
                    info["valid_step"] = 0.0
                    info["risk"] = 1.0
            elif action.type == ActionType.ANSWER:
                child.final_answer = action.content.strip()
                if not child.final_answer:
                    info["goal_progress"] = 0.0
                    info["proof_completion"] = 0.0
                    info["valid_step"] = 0.0
                    info["risk"] = 1.0
                    info["note"] = "empty answer"
                elif self.answer_judge(child, child.final_answer):
                    child.status = "solved"
                    info["goal_progress"] = 1.0
                    info["proof_completion"] = 1.0
                else:
                    info["goal_progress"] = 0.0
                    info["proof_completion"] = 0.0
                    info["valid_step"] = 0.25
                    info["risk"] = 0.75
                    info["note"] = "wrong answer"
            elif action.type == ActionType.BACKTRACK:
                info["goal_progress"] = -0.05
                info["risk"] = 0.15
            else:
                info["valid_step"] = 0.0
                info["risk"] = 1.0
                info["note"] = "unsupported action"
        except Exception as exc:
            info["valid_step"] = 0.0
            info["risk"] = 1.0
            info["note"] = f"exception: {exc}"

        if child.status == "solved":
            info["proof_completion"] = 1.0
        return child, info
