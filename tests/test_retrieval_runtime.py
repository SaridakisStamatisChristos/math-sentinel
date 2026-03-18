from __future__ import annotations

import unittest

from memory.retrieval import retrieve_context
from sentinel.search_runtime import build_action_bias_fn


class _LemmaStore:
    def retrieve(self, domain: str, text: str, limit: int = 5):  # noqa: ARG002
        return []


class _HardCaseStore:
    def retrieve(self, domain: str, limit: int = 5):  # noqa: ARG002
        return [
            {"task": "inspect_workspace then apply_patch", "answer": "wrong_patch", "expected": "patched_and_verified", "domain": "swebench_patch"},
        ]


class RetrievalRuntimeTests(unittest.TestCase):
    def test_retrieve_context_supports_none_mode(self) -> None:
        context = retrieve_context(
            _LemmaStore(),
            _HardCaseStore(),
            "swebench_patch",
            "Patch the repository carefully",
            mode="none",
            tool_names=["apply_patch"],
        )

        self.assertEqual(context["lemmas"], [])
        self.assertEqual(context["tool_priors"], {})

    def test_retrieve_context_surfaces_tool_priors_and_failure_avoidance(self) -> None:
        context = retrieve_context(
            _LemmaStore(),
            _HardCaseStore(),
            "swebench_patch",
            "Patch the repository and apply_patch carefully",
            tool_names=["inspect_workspace", "apply_patch", "run_unit_tests"],
        )

        self.assertIn("apply_patch", context["tool_priors"])
        self.assertGreaterEqual(len(context["failure_avoidance"]), 1)

    def test_action_bias_uses_retrieval_context(self) -> None:
        class _Action:
            tool = "apply_patch"
            content = ""

            class type:
                value = "APPLY"

        class _State:
            domain = "swebench_patch"
            metadata = {"_retrieval_context": {"tool_priors": {"apply_patch": 1.0}, "failure_avoidance": []}}

        bias = build_action_bias_fn(None)

        self.assertGreater(bias(_State(), _Action()), 0.5)
