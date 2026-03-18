
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Tuple


WORD_RE = re.compile(r"[a-zA-Z_]+")


class HashingTextEncoder:
    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def encode(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for word in WORD_RE.findall(text.lower()):
            index = hash(word) % self.dim
            vec[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vec)) or 1.0
        return [value / norm for value in vec]


def _cosine(left: Iterable[float], right: Iterable[float]) -> float:
    left_list = list(left)
    right_list = list(right)
    return sum(a * b for a, b in zip(left_list, right_list))


def _item_text(item: Any) -> str:
    if hasattr(item, "pattern"):
        return f"{getattr(item, 'pattern', '')} {getattr(item, 'statement', '')}"
    if isinstance(item, dict):
        return " ".join(str(item.get(key, "")) for key in ["task", "answer", "expected", "domain"])
    return str(item)


class RetrievalService:
    def __init__(self, mode: str = "hybrid", embedding_model: str = "hashing") -> None:
        self.mode = mode
        self.embedding_model = embedding_model
        self.encoder = HashingTextEncoder()

    def rerank(self, query: str, items: List[Any], limit: int) -> List[Any]:
        if self.mode == "lexical" or not items:
            return items[:limit]
        query_vec = self.encoder.encode(query)
        scored: List[Tuple[float, Any]] = []
        for item in items:
            text = _item_text(item)
            base = 0.0
            if self.mode in {"hybrid", "embedding"}:
                base = _cosine(query_vec, self.encoder.encode(text))
            scored.append((base, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]


def retrieve_context(
    lemma_store: Any,
    hard_case_store: Any,
    domain: str,
    text: str,
    mode: str = "hybrid",
    embedding_model: str = "hashing",
    event_logger: Any | None = None,
) -> Dict[str, List[Any]]:
    service = RetrievalService(mode=mode, embedding_model=embedding_model)
    lemmas = lemma_store.retrieve(domain, text, limit=6)
    hard_cases = hard_case_store.retrieve(domain, limit=6)
    result = {
        "lemmas": service.rerank(text, lemmas, limit=3),
        "hard_cases": service.rerank(text, hard_cases, limit=3),
    }
    if event_logger is not None:
        event_type = "retrieval_hit" if result["lemmas"] or result["hard_cases"] else "retrieval_miss"
        event_logger(
            event_type,
            domain=domain,
            query=text[:160],
            mode=mode,
            embedding_model=embedding_model,
            lemma_hits=len(result["lemmas"]),
            hard_case_hits=len(result["hard_cases"]),
        )
    return result
