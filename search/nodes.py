
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from proof.state import ProofState
from proof.actions import Action


@dataclass
class SearchNode:
    state: ProofState
    cumulative_score: float
    local_scores: Dict[str, float] = field(default_factory=dict)
    parent: Optional["SearchNode"] = None
    action: Optional[Action] = None
    depth: int = 0
