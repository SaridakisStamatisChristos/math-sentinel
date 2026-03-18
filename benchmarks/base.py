from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class BenchmarkSuite:
    name: str
    backend: str
    description: str
    tier: str
    cases: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkCaseResult:
    task_id: str
    domain: str
    prompt: str
    expected_answer: str
    final_answer: str
    status: str
    solved: bool
    equivalent: bool
    explored_nodes: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkSuiteResult:
    suite: str
    backend: str
    tier: str
    description: str
    solved_rate: float
    equivalence_rate: float
    avg_branches: float
    cases: List[BenchmarkCaseResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["cases"] = [case.to_dict() for case in self.cases]
        return payload
