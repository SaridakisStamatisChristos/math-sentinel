from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from .actions import Action, ActionType
from .goals import add_subgoal, resolve_subgoal
from .state import ReasoningState
from .tool_results import normalize_tool_result


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
                self._bump_stagnation(child, 0.0)
            elif action.type == ActionType.SUBGOAL:
                add_subgoal(child, action.content)
                info["goal_progress"] = 0.1
                self._bump_stagnation(child, info["goal_progress"])
            elif action.type == ActionType.RESOLVE_SUBGOAL:
                resolve_subgoal(child, action.content)
                info["goal_progress"] = 0.2
                self._resolve_obligations(child, [action.content])
                self._bump_stagnation(child, info["goal_progress"])
            elif action.type == ActionType.LEMMA:
                child.lemma_refs.append(action.content.strip() or action.name or "anonymous_lemma")
                info["goal_progress"] = 0.05
                self._bump_stagnation(child, info["goal_progress"])
            elif action.type == ActionType.ASSUME:
                child.assumptions.append(action.content)
                info["goal_progress"] = 0.02
                self._bump_stagnation(child, info["goal_progress"])
            elif action.type in {ActionType.APPLY, ActionType.CALL_PLUGIN, ActionType.CHECK, ActionType.REWRITE, ActionType.SIMPLIFY}:
                result = normalize_tool_result(self.tool_registry.call(action.tool, action.content, child))
                child.tool_history.append({"tool": action.tool, "input": action.content, "result": result})
                if result.get("ok"):
                    rendered = result.get("result_text", "")
                    if rendered:
                        child.derived_facts.append(str(rendered))
                        child.fact_provenance.append({"fact": str(rendered), "tool": action.tool, "input": action.content})
                    payload = result.get("result_payload", {})
                    if payload:
                        child.tool_payloads.append({"tool": action.tool, "payload": payload})
                        self._extend_unique(child.dependency_refs, (str(dep) for dep in payload.get("dependencies", [])))
                        self._extend_unique(child.obligations, (str(item) for item in payload.get("obligations", [])))
                        self._extend_unique(child.evidence_refs, (str(item) for item in payload.get("evidence", [])))
                        self._resolve_obligations(child, (str(item) for item in payload.get("resolved_obligations", [])))
                        suggested_tools = [str(item) for item in payload.get("suggested_tools", []) if str(item).strip()]
                        if suggested_tools:
                            child.metadata["suggested_tools"] = suggested_tools
                    info["goal_progress"] = float(result.get("goal_progress", 0.2))
                    info["risk"] = float(result.get("risk", 0.0))
                    if payload.get("candidate_answer") and not child.final_answer:
                        child.metadata["candidate_answer"] = str(payload["candidate_answer"])
                    if result.get("solved"):
                        child.final_answer = str(result.get("answer", rendered))
                        child.status = "solved"
                        child.terminal_confidence = max(child.terminal_confidence, 1.0 - info["risk"])
                    self._track_tool_usage(child, action.tool, action.content)
                    self._bump_stagnation(child, info["goal_progress"])
                else:
                    info["valid_step"] = 0.0
                    info["risk"] = max(1.0, float(result.get("risk", 1.0)))
                    self._track_tool_usage(child, action.tool, action.content)
                    self._bump_stagnation(child, 0.0)
            elif action.type == ActionType.ANSWER:
                child.final_answer = action.content.strip()
                if not child.final_answer:
                    info["goal_progress"] = 0.0
                    info["proof_completion"] = 0.0
                    info["valid_step"] = 0.0
                    info["risk"] = 1.0
                    info["note"] = "empty answer"
                    self._bump_stagnation(child, 0.0)
                elif self.answer_judge(child, child.final_answer):
                    child.status = "solved"
                    info["goal_progress"] = 1.0
                    info["proof_completion"] = 1.0
                    child.terminal_confidence = max(child.terminal_confidence, 1.0)
                    self._resolve_obligations(child, list(child.obligations))
                    self._bump_stagnation(child, info["goal_progress"])
                else:
                    info["goal_progress"] = 0.0
                    info["proof_completion"] = 0.0
                    info["valid_step"] = 0.25
                    info["risk"] = 0.75
                    info["note"] = "wrong answer"
                    self._bump_stagnation(child, 0.0)
            elif action.type == ActionType.BACKTRACK:
                info["goal_progress"] = -0.05
                info["risk"] = 0.15
                self._bump_stagnation(child, 0.0)
            else:
                info["valid_step"] = 0.0
                info["risk"] = 1.0
                info["note"] = "unsupported action"
                self._bump_stagnation(child, 0.0)
        except Exception as exc:
            info["valid_step"] = 0.0
            info["risk"] = 1.0
            info["note"] = f"exception: {exc}"
            self._bump_stagnation(child, 0.0)

        if child.status == "solved":
            info["proof_completion"] = 1.0
            child.terminal_confidence = max(child.terminal_confidence, 1.0 - float(info.get("risk", 0.0)))
        return child, info

    @staticmethod
    def _extend_unique(target: list[str], values: Any) -> None:
        existing = set(target)
        for value in values:
            text = str(value).strip()
            if text and text not in existing:
                target.append(text)
                existing.add(text)

    @staticmethod
    def _resolve_obligations(state: ReasoningState, values: Any) -> None:
        resolved = {str(value).strip() for value in values if str(value).strip()}
        if not resolved:
            return
        state.obligations = [item for item in state.obligations if item not in resolved]

    @staticmethod
    def _track_tool_usage(state: ReasoningState, tool: str, content: str) -> None:
        current = f"{tool}::{content.strip()}"
        previous = str(state.metadata.get("last_tool_signature", ""))
        repeated = int(state.metadata.get("repeat_tool_steps", 0))
        if current and current == previous:
            repeated += 1
        else:
            repeated = 0
        state.metadata["last_tool_signature"] = current
        state.metadata["repeat_tool_steps"] = repeated

    @staticmethod
    def _bump_stagnation(state: ReasoningState, goal_progress: float) -> None:
        streak = int(state.metadata.get("no_progress_streak", 0))
        if goal_progress > 0.0:
            streak = 0
        else:
            streak += 1
        state.metadata["no_progress_streak"] = streak
