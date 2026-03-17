from __future__ import annotations

import unittest

from domains.code_ops.backend import CodeOpsReasoningDomain


class CodeOpsBackendTests(unittest.TestCase):
    def test_manual_function_name_task_solves_via_fallback_tool(self) -> None:
        backend = CodeOpsReasoningDomain()
        task = backend.manual_task(
            "function_name",
            "Read the Python function and return the function name:\ndef helper(x):\n    return x + 1",
        )
        state = backend.make_state(task)
        repair = backend.fallback_repairs(state)[0]

        child, info = backend.create_executor().apply(state, repair)

        self.assertEqual(child.status, "solved")
        self.assertEqual(child.final_answer, "helper")
        self.assertGreater(info["goal_progress"], 0.0)

    def test_has_loop_answer_is_case_insensitive(self) -> None:
        backend = CodeOpsReasoningDomain()
        task = backend.manual_task(
            "has_loop",
            "Does this Python function contain a loop? Return yes or no.\ndef f(x):\n    for item in range(3):\n        print(item)\n    return x",
            "yes",
        )

        self.assertTrue(backend.evaluate_answer(task, "YES"))
        self.assertFalse(backend.evaluate_answer(task, "no"))


if __name__ == "__main__":
    unittest.main()
