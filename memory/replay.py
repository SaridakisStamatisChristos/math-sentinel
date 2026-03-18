
from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List


class ReplayBuffer:
    def __init__(self, capacity: int = 5000) -> None:
        self.capacity = capacity
        self.items: Deque[Dict[str, Any]] = deque(maxlen=capacity)

    def add(self, item: Dict[str, Any]) -> None:
        self.items.append(item)

    def sample(self, k: int) -> List[Dict[str, Any]]:
        k = min(k, len(self.items))
        return random.sample(list(self.items), k) if k > 0 else []

    def sample_weighted(self, k: int) -> List[Dict[str, Any]]:
        items = list(self.items)
        k = min(k, len(items))
        if k <= 0:
            return []
        weights = [float(item.get("weight", item.get("score", 1.0) or 1.0)) for item in items]
        total = sum(max(weight, 0.01) for weight in weights)
        normalized = [max(weight, 0.01) / total for weight in weights]
        chosen: List[Dict[str, Any]] = []
        available_items = items[:]
        available_weights = normalized[:]
        for _ in range(k):
            index = random.choices(range(len(available_items)), weights=available_weights, k=1)[0]
            chosen.append(available_items.pop(index))
            available_weights.pop(index)
            if not available_items:
                break
        return chosen

    def save_jsonl(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in self.items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def load_jsonl(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        items: Deque[Dict[str, Any]] = deque(maxlen=self.capacity)
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        self.items = items
