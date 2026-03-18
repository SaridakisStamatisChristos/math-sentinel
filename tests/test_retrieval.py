from __future__ import annotations

import unittest

from memory.retrieval import HashingTextEncoder, RetrievalService, retrieve_context


class _Lemma:
    def __init__(self, pattern: str, statement: str) -> None:
        self.pattern = pattern
        self.statement = statement


class _LemmaStore:
    def retrieve(self, domain: str, text: str, limit: int = 5) -> list[_Lemma]:
        return [
            _Lemma("project plan", "Plan with dependencies"),
            _Lemma("factorization", "Math lemma"),
        ]


class _HardCaseStore:
    def retrieve(self, domain: str, limit: int = 5) -> list[dict[str, object]]:
        return [
            {"task": "project plan under budget", "domain": domain},
            {"task": "factorization edge case", "domain": domain},
        ]


class RetrievalTests(unittest.TestCase):
    def test_hashing_text_encoder_is_stable_shape(self) -> None:
        encoder = HashingTextEncoder(dim=16)
        vec = encoder.encode("project plan with dependencies")

        self.assertEqual(len(vec), 16)

    def test_retrieval_context_returns_reranked_items(self) -> None:
        ctx = retrieve_context(_LemmaStore(), _HardCaseStore(), "planning_ops", "project plan with budget")

        self.assertEqual(len(ctx["lemmas"]), 2)
        self.assertEqual(len(ctx["hard_cases"]), 2)

    def test_retrieval_service_reranks_embedding_mode(self) -> None:
        service = RetrievalService(mode="embedding")
        items = ["project plan", "factorization task"]

        ranked = service.rerank("project plan", items, limit=2)

        self.assertEqual(ranked[0], "project plan")


if __name__ == "__main__":
    unittest.main()
