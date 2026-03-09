
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict


class TacticStats:
    def __init__(self) -> None:
        self.counts: DefaultDict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record(self, domain: str, tactic: str, success: bool) -> None:
        key = f"{tactic}::{'ok' if success else 'fail'}"
        self.counts[domain][key] += 1

    def bias(self, domain: str, tactic: str) -> float:
        ok = self.counts[domain].get(f"{tactic}::ok", 0)
        fail = self.counts[domain].get(f"{tactic}::fail", 0)
        total = ok + fail
        return 0.5 if total == 0 else ok / total

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.counts, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        counts: DefaultDict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for domain, stats in data.items():
            counts[domain].update({key: int(value) for key, value in stats.items()})
        self.counts = counts
