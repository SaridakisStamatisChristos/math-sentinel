from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import OrderedDict as OrderedDictType, Tuple


@dataclass
class TranspositionRecord:
    best_score: float
    best_depth: int
    visits: int = 1


class TranspositionTable:
    def __init__(self, capacity: int = 4096) -> None:
        self.capacity = max(1, int(capacity))
        self._records: OrderedDictType[str, TranspositionRecord] = OrderedDict()

    def register(self, signature: str, score: float, depth: int) -> Tuple[bool, float]:
        record = self._records.get(signature)
        if record is None:
            self._records[signature] = TranspositionRecord(best_score=score, best_depth=depth)
            self._trim()
            return True, 1.0

        record.visits += 1
        novelty = 1.0 / float(record.visits)
        improved = score > record.best_score + 1e-6 or depth < record.best_depth
        if improved:
            record.best_score = max(record.best_score, score)
            record.best_depth = min(record.best_depth, depth)
            self._records.move_to_end(signature)
            return True, novelty
        return False, novelty

    def _trim(self) -> None:
        while len(self._records) > self.capacity:
            self._records.popitem(last=False)
