
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class HardCaseStore:
    def __init__(self, capacity: int = 2000) -> None:
        self.capacity = capacity
        self.cases: List[Dict] = []

    def add(self, case: Dict) -> None:
        self.cases.append(case)
        if len(self.cases) > self.capacity:
            self.cases = self.cases[-self.capacity :]

    def retrieve(self, domain: str, limit: int = 5) -> List[Dict]:
        items = [c for c in self.cases if c.get("domain") == domain]
        items.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return items[:limit]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.cases, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        self.cases = list(data)[-self.capacity :]
