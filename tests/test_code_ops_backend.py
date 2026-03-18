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

    def test_assignment_count_and_distinct_calls_solve_via_fallback_tools(self) -> None:
        backend = CodeOpsReasoningDomain()
        executor = backend.create_executor()

        assignment_task = backend.manual_task(
            "assignment_count",
            "Count assignment statements in this Python function:\ndef build():\n    x = 1\n    y = 2\n    return x + y",
        )
        assign_state = backend.make_state(assignment_task)
        assign_child, _ = executor.apply(assign_state, backend.fallback_repairs(assign_state)[0])
        self.assertEqual(assign_child.final_answer, "2")
        self.assertEqual(assign_child.status, "solved")

        calls_task = backend.manual_task(
            "called_function_count",
            "Count distinct called functions in this Python function:\ndef dispatch(x):\n    helper(x)\n    format_item(x)\n    helper(x)\n    return x",
        )
        calls_state = backend.make_state(calls_task)
        calls_child, _ = executor.apply(calls_state, backend.fallback_repairs(calls_state)[0])
        self.assertEqual(calls_child.final_answer, "2")
        self.assertEqual(calls_child.status, "solved")

    def test_repo_patch_task_solves_via_multi_step_fallback_loop(self) -> None:
        backend = CodeOpsReasoningDomain()
        state = backend.make_state(backend.benchmark_tasks()[0])
        executor = backend.create_executor()

        for _ in range(6):
            repair = backend.fallback_repairs(state)[0]
            state, _ = executor.apply(state, repair)
            if state.final_answer:
                answer_action = backend.fallback_repairs(state)[0]
                state, _ = executor.apply(state, answer_action)
            if state.status == "solved":
                break

        self.assertEqual(state.status, "solved")
        self.assertEqual(state.final_answer, "patched_and_verified")
        self.assertIn("app.py", "\n".join(state.metadata.get("workspace_files", [])))


if __name__ == "__main__":
    unittest.main()
