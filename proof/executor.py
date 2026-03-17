from __future__ import annotations

from typing import Any

from engine.executor import StateExecutor

from .equivalence import equivalent
from .state import ProofState


class ProofExecutor(StateExecutor):
    def __init__(self, tool_registry: Any) -> None:
        super().__init__(tool_registry=tool_registry, answer_judge=self._answer_judge)

    @staticmethod
    def _answer_judge(state: ProofState, candidate: str) -> bool:
        return equivalent(state.domain, candidate, state.expected_answer, state.metadata)
