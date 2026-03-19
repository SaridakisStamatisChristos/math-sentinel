from __future__ import annotations

import unittest

from benchmark_v7 import default_campaign_profile_for_suite
from benchmarks.runners import _case_audit_from_search
from engine.state import ReasoningState
from engine.task import ReasoningTask
from search.nodes import SearchNode


class BenchmarkIntegrityTests(unittest.TestCase):
    def test_default_public_campaign_profile_is_strict(self) -> None:
        self.assertEqual(default_campaign_profile_for_suite("public_smoke"), "public_claim_no_repairs")
        self.assertEqual(default_campaign_profile_for_suite("public_medium"), "public_claim_no_repairs")
        self.assertEqual(default_campaign_profile_for_suite("internal"), "smoke_tiny")

    def test_case_audit_detects_runtime_oracle_metadata(self) -> None:
        task = ReasoningTask(
            task_id="gaia_case",
            domain="gaia_csv_reasoning",
            prompt="question",
            answer="42",
            goal="answer",
            meta={"oracle_evidence_file": "sales.csv"},
        )
        state = ReasoningState(
            task_id="gaia_case",
            domain="gaia_csv_reasoning",
            problem_text="question",
            goal="answer",
            expected_answer="42",
            metadata={
                "benchmark_assistance_mode": "unassisted",
                "oracle_evidence_file": "sales.csv",
            },
        )
        explored = [SearchNode(state=state, cumulative_score=0.0, local_scores={"fallback_repair_used": 1.0, "guided_rollout_used": 1.0}, depth=1)]

        audit = _case_audit_from_search({"benchmark": {"assistance_mode": "unassisted"}}, task, state, explored)

        self.assertFalse(audit["benchmark_integrity_passed"])
        self.assertTrue(audit["oracle_fields_present_in_runtime"])
        self.assertTrue(audit["guided_rollout_used"])
        self.assertTrue(audit["fallback_repair_used"])


if __name__ == "__main__":
    unittest.main()
