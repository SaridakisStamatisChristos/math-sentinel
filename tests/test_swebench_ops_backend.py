from __future__ import annotations

import unittest

from domains.swebench_ops.backend import SwebenchOpsReasoningDomain


class SwebenchOpsBackendTests(unittest.TestCase):
    def test_gold_patch_repair_solves_fixture_case(self) -> None:
        backend = SwebenchOpsReasoningDomain()
        task = backend.benchmark_tasks()[0]
        state = backend.make_state(task)
        executor = backend.create_executor()

        repair = next(action for action in backend.fallback_repairs(state) if action.tool == "apply_gold_patch")
        child, info = executor.apply(state, repair)

        self.assertEqual(child.status, "solved")
        self.assertEqual(child.final_answer, "patched_and_verified")
        self.assertGreaterEqual(float(info["goal_progress"]), 0.8)

    def test_action_schema_exposes_patch_tools(self) -> None:
        backend = SwebenchOpsReasoningDomain()
        task = backend.benchmark_tasks()[0]
        state = backend.make_state(task)
        schema = backend.action_schema(state)

        self.assertTrue(schema["strict"])
        self.assertIn("APPLY", schema["action_types"])
        self.assertIn("apply_gold_patch", schema["action_types"]["APPLY"]["tools"])

    def test_manual_task_matches_fixture_case_for_sample_flow(self) -> None:
        backend = SwebenchOpsReasoningDomain()

        task = backend.manual_task(
            "swebench_patch",
            "Patch the repository so the failing tests pass. Fix the arithmetic bug in app.py and verify with the test suite.",
        )

        self.assertIn("fixture_dir", task.meta)
        state = backend.make_state(task)
        self.assertIn("app.py", state.metadata["workspace_files"])
