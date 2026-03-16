
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class HardCaseStore:
    def __init__(self, capacity: int = 2000) -> None:
        self.capacity = capacity
        self.cases: List[Dict] = []

    def add(self, case: Dict) -> None:
        sanitized = dict(case)
        # Never persist an empty answer; fallback to expected or placeholder for easier debugging.
        if not sanitized.get("answer"):
            sanitized["answer"] = sanitized.get("expected") or "<no_answer>"
        self.cases.append(sanitized)
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
        cleaned = []
        for c in data:
            if not c.get("answer"):
                c["answer"] = c.get("expected") or "<no_answer>"
            cleaned.append(c)
        self.cases = cleaned[-self.capacity :]
