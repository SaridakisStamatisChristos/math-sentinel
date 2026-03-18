from __future__ import annotations

import unittest

from domains.swebench_ops.backend import SwebenchOpsReasoningDomain


class SwebenchOpsBackendTests(unittest.TestCase):
    def test_fallback_loop_solves_fixture_case_without_oracle_patch_hints(self) -> None:
        backend = SwebenchOpsReasoningDomain()
        task = backend.benchmark_tasks()[0]
        task.meta.pop("oracle_primary_file", None)
        state = backend.make_state(task)
        executor = backend.create_executor()

        info = {}
        for _ in range(10):
            repair = backend.fallback_repairs(state)[0]
            state, info = executor.apply(state, repair)
            if state.final_answer:
                answer_action = backend.fallback_repairs(state)[0]
                state, info = executor.apply(state, answer_action)
            if state.status == "solved":
                break

        self.assertEqual(state.status, "solved")
        self.assertEqual(state.final_answer, "patched_and_verified")
        self.assertGreaterEqual(float(info["goal_progress"]), 0.8)
        self.assertEqual(state.metadata.get("benchmark_assistance_mode"), "unassisted")

    def test_action_schema_exposes_patch_tools(self) -> None:
        backend = SwebenchOpsReasoningDomain()
        task = backend.benchmark_tasks()[0]
        state = backend.make_state(task)
        schema = backend.action_schema(state)

        self.assertTrue(schema["strict"])
        self.assertIn("APPLY", schema["action_types"])
        self.assertIn("inspect_tests", schema["action_types"]["APPLY"]["tools"])
        self.assertIn("localize_failure", schema["action_types"]["APPLY"]["tools"])
        self.assertIn("draft_patch", schema["action_types"]["APPLY"]["tools"])
        self.assertNotIn("apply_gold_patch", schema["action_types"]["APPLY"]["tools"])

    def test_manual_task_matches_fixture_case_for_sample_flow(self) -> None:
        backend = SwebenchOpsReasoningDomain()

        task = backend.manual_task(
            "swebench_patch",
            "Patch the repository so the failing tests pass. Fix the arithmetic bug in app.py and verify with the test suite.",
        )

        self.assertIn("fixture_dir", task.meta)
        state = backend.make_state(task)
        self.assertIn("app.py", state.metadata["workspace_files"])

    def test_unassisted_runtime_state_strips_oracle_metadata(self) -> None:
        backend = SwebenchOpsReasoningDomain()
        task = backend.benchmark_tasks()[0]

        state = backend.make_state(task)

        self.assertNotIn("oracle_primary_file", state.metadata)
        self.assertNotIn("oracle_patch", state.metadata)
        self.assertEqual(state.metadata.get("benchmark_audit", {}).get("assistance_mode"), "unassisted")

    def test_medium_fixture_fallback_loop_solves_argument_order_bug(self) -> None:
        backend = SwebenchOpsReasoningDomain()
        task = next(task for task in backend.benchmark_tasks() if task.task_id == "swebench_argument_order_bug")
        state = backend.make_state(task)
        executor = backend.create_executor()

        for _ in range(10):
            repair = backend.fallback_repairs(state)[0]
            state, _ = executor.apply(state, repair)
            if state.final_answer:
                answer_action = backend.fallback_repairs(state)[0]
                state, _ = executor.apply(state, answer_action)
            if state.status == "solved":
                break

        self.assertEqual(state.status, "solved")
        self.assertEqual(state.final_answer, "patched_and_verified")
