from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ReasoningTask:
    task_id: str
    domain: str
    prompt: str
    answer: str
    goal: str
    meta: Dict[str, Any]
