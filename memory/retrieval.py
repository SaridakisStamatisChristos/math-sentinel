
from __future__ import annotations

from typing import Any, Dict, List


def retrieve_context(lemma_store: Any, hard_case_store: Any, domain: str, text: str) -> Dict[str, List[Any]]:
    return {
        "lemmas": lemma_store.retrieve(domain, text, limit=3),
        "hard_cases": hard_case_store.retrieve(domain, limit=3),
    }
