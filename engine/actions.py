from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class ActionType(str, Enum):
    THINK = "THINK"
    APPLY = "APPLY"
    CHECK = "CHECK"
    ANSWER = "ANSWER"
    REWRITE = "REWRITE"
    LEMMA = "LEMMA"
    SUBGOAL = "SUBGOAL"
    RESOLVE_SUBGOAL = "RESOLVE_SUBGOAL"
    ASSUME = "ASSUME"
    BACKTRACK = "BACKTRACK"
    CALL_PLUGIN = "CALL_PLUGIN"
    SIMPLIFY = "SIMPLIFY"


@dataclass
class Action:
    type: ActionType
    content: str = ""
    tool: str = ""
    name: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        if self.type in {ActionType.APPLY, ActionType.CALL_PLUGIN} and not self.tool:
            return False
        if self.type == ActionType.ANSWER and not self.content.strip():
            return False
        return True

    def to_record(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "content": self.content,
            "tool": self.tool,
            "name": self.name,
            "payload": self.payload,
        }
