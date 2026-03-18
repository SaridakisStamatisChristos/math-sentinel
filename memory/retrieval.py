
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple


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


def _lexical_overlap(query: str, text: str) -> float:
    query_terms = set(WORD_RE.findall(query.lower()))
    text_terms = set(WORD_RE.findall(text.lower()))
    if not query_terms or not text_terms:
        return 0.0
    return len(query_terms & text_terms) / float(len(query_terms | text_terms))


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
            ranked = sorted(items, key=lambda item: _lexical_overlap(query, _item_text(item)), reverse=True)
            return ranked[:limit]
        query_vec = self.encoder.encode(query)
        scored: List[Tuple[float, Any]] = []
        for item in items:
            text = _item_text(item)
            base = 0.0
            if self.mode in {"hybrid", "embedding"}:
                base = _cosine(query_vec, self.encoder.encode(text))
            if self.mode in {"hybrid", "lexical"}:
                base += 0.35 * _lexical_overlap(query, text)
            scored.append((base, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]


def _collect_tool_priors(
    query: str,
    items: Sequence[Any],
    tool_names: Sequence[str],
) -> Dict[str, float]:
    if not tool_names:
        return {}
    counter: Counter[str] = Counter()
    query_terms = set(WORD_RE.findall(query.lower()))
    for item in items:
        text = _item_text(item).lower()
        for tool in tool_names:
            tool_name = str(tool).strip().lower()
            if not tool_name:
                continue
            if tool_name in text:
                counter[tool_name] += 3
                continue
            parts = [part for part in tool_name.split("_") if part]
            overlap = sum(1 for part in parts if part in text or part in query_terms)
            if overlap:
                counter[tool_name] += overlap
    total = float(sum(counter.values())) or 1.0
    return {tool: score / total for tool, score in counter.items()}


def _collect_failure_avoidance(items: Sequence[Any]) -> List[str]:
    avoid: List[str] = []
    seen: set[str] = set()
    for item in items:
        text = _item_text(item).strip()
        if not text:
            continue
        lowered = text.lower()
        message = ""
        if "wrong" in lowered or "fail" in lowered:
            message = text[:120]
        elif isinstance(item, dict) and (item.get("expected") and item.get("answer") and item.get("expected") != item.get("answer")):
            message = f"avoid previous mismatch: expected {item.get('expected')} not {item.get('answer')}"
        if message and message not in seen:
            avoid.append(message)
            seen.add(message)
    return avoid[:3]


def retrieve_context(
    lemma_store: Any,
    hard_case_store: Any,
    domain: str,
    text: str,
    mode: str = "hybrid",
    embedding_model: str = "hashing",
    tool_names: Sequence[str] | None = None,
    event_logger: Any | None = None,
) -> Dict[str, Any]:
    service = RetrievalService(mode=mode, embedding_model=embedding_model)
    lemmas = lemma_store.retrieve(domain, text, limit=6)
    hard_cases = hard_case_store.retrieve(domain, limit=6)
    ranked_lemmas = service.rerank(text, lemmas, limit=3)
    ranked_hard_cases = service.rerank(text, hard_cases, limit=3)
    tool_priors = _collect_tool_priors(text, list(ranked_lemmas) + list(ranked_hard_cases), tool_names or [])
    result = {
        "lemmas": ranked_lemmas,
        "hard_cases": ranked_hard_cases,
        "tool_priors": tool_priors,
        "failure_avoidance": _collect_failure_avoidance(ranked_hard_cases),
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
            tool_hints=",".join(sorted(result["tool_priors"].keys())),
        )
    return result
