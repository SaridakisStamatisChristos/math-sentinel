from __future__ import annotations

import unittest

from domains.string_ops.backend import StringOpsReasoningDomain


class StringOpsBackendTests(unittest.TestCase):
    def test_manual_sort_task_solves_via_fallback_tool(self) -> None:
        backend = StringOpsReasoningDomain()
        task = backend.manual_task("sort_words", "Sort words alphabetically: kiwi apple mango")
        state = backend.make_state(task)
        repair = backend.fallback_repairs(state)[0]

        child, info = backend.create_executor().apply(state, repair)

        self.assertEqual(child.status, "solved")
        self.assertEqual(child.final_answer, "apple kiwi mango")
        self.assertGreater(info["goal_progress"], 0.0)

    def test_evaluate_answer_normalizes_whitespace_for_sequence_tasks(self) -> None:
        backend = StringOpsReasoningDomain()
        task = backend.manual_task("dedupe_words", "Remove duplicate words preserving order: red blue red green blue", "red blue green")

        self.assertTrue(backend.evaluate_answer(task, "red   blue   green"))
        self.assertFalse(backend.evaluate_answer(task, "red green blue"))


if __name__ == "__main__":
    unittest.main()
