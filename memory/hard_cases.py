
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


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

    def add_failure_bundle(self, case: Dict) -> None:
        bundle = dict(case)
        bundle.setdefault("source", "benchmark")
        self.add(bundle)

    def retrieve(self, domain: str, limit: int = 5, filters: Dict[str, Any] | None = None) -> List[Dict]:
        filters = dict(filters or {})
        exclude_sources = {str(item) for item in filters.get("exclude_sources", []) if str(item).strip()}
        exclude_suites = {str(item) for item in filters.get("exclude_suites", []) if str(item).strip()}
        exclude_holdout_groups = {str(item) for item in filters.get("exclude_holdout_groups", []) if str(item).strip()}
        exclude_task_ids = {str(item) for item in filters.get("exclude_task_ids", []) if str(item).strip()}
        items = [c for c in self.cases if c.get("domain") == domain]
        if exclude_sources or exclude_suites or exclude_holdout_groups or exclude_task_ids:
            filtered: List[Dict] = []
            for case in items:
                source = str(case.get("source", "")).strip()
                suite = str(case.get("suite", "")).strip()
                holdout_group = str(case.get("holdout_group", "")).strip()
                task_id = str(case.get("task_id", "")).strip()
                if source and source in exclude_sources:
                    continue
                if suite and suite in exclude_suites:
                    continue
                if holdout_group and holdout_group in exclude_holdout_groups:
                    continue
                if task_id and task_id in exclude_task_ids:
                    continue
                filtered.append(case)
            items = filtered
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
