from __future__ import annotations

from memory.hard_cases import HardCaseStore


def test_hard_case_store_injects_placeholder_answer() -> None:
    store = HardCaseStore(capacity=3)
    store.add({"task": "t", "domain": "d", "score": 1.0, "answer": ""})

    case = store.cases[-1]
    assert case["answer"] == "<no_answer>"
