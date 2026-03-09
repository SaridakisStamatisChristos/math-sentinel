
from __future__ import annotations

from typing import Dict, Any

from proof.equivalence import equivalent
from .generators import GeneratedTask


def evaluate_answer(task: GeneratedTask, candidate: str) -> bool:
    return equivalent(task.domain, candidate, task.answer, task.meta)


def target_metadata(task: GeneratedTask) -> Dict[str, Any]:
    return {"family": task.meta.get("family", task.domain)}
