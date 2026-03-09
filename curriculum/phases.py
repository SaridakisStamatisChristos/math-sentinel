
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Phase:
    name: str
    until_step: int
    domains: List[str]


class PhaseScheduler:
    def __init__(self, phases: List[Phase]) -> None:
        self.phases = phases

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "PhaseScheduler":
        phases = [Phase(name=p["name"], until_step=int(p["until_step"]), domains=list(p["domains"])) for p in cfg["phases"]]
        return cls(phases)

    def phase_for_step(self, step: int) -> Phase:
        for phase in self.phases:
            if step <= phase.until_step:
                return phase
        return self.phases[-1]
