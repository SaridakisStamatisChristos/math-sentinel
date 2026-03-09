
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ProofState:
    task_id: str
    domain: str
    problem_text: str
    goal: str
    expected_answer: str = ""
    assumptions: List[str] = field(default_factory=list)
    derived_facts: List[str] = field(default_factory=list)
    subgoals: List[str] = field(default_factory=list)
    lemma_refs: List[str] = field(default_factory=list)
    tool_history: List[Dict[str, Any]] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "open"
    final_answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "ProofState":
        return copy.deepcopy(self)

    def serialize(self) -> str:
        lines = [
            f"[DOMAIN] {self.domain}",
            f"[PROBLEM] {self.problem_text}",
            f"[GOAL] {self.goal}",
            f"[ASSUMPTIONS] {' | '.join(self.assumptions) if self.assumptions else 'none'}",
            f"[DERIVED] {' | '.join(self.derived_facts) if self.derived_facts else 'none'}",
            f"[SUBGOALS] {' | '.join(self.subgoals) if self.subgoals else 'none'}",
            f"[LEMMAS] {' | '.join(self.lemma_refs) if self.lemma_refs else 'none'}",
            f"[TOOL_HISTORY] {self._stringify(self.tool_history)}",
            f"[ACTION_HISTORY] {self._stringify(self.action_history)}",
            f"[STATUS] {self.status}",
            f"[FINAL_ANSWER] {self.final_answer if self.final_answer else 'none'}",
            f"[METADATA] {self.metadata if self.metadata else {}}",
            "[END_STATE]",
        ]
        return "\n".join(lines)

    def short_problem(self) -> str:
        return self.problem_text[:120]

    @staticmethod
    def _stringify(items: List[Any]) -> str:
        if not items:
            return "none"
        return " || ".join(str(x) for x in items[-6:])
