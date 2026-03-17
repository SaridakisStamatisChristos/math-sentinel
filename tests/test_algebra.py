from __future__ import annotations

import unittest

from tools.algebra import solve_linear_step
from tools.arithmetic import divmod_tool


class AlgebraToolTests(unittest.TestCase):
    def test_solve_linear_step_handles_plus_minus_constant(self) -> None:
        # Exercise the regex fallback by forcing sympy off.
        import tools.algebra as algebra

        original_sp = algebra.sp
        algebra.sp = None
        try:
            result = solve_linear_step("2x + -4 = 0")
        finally:
            algebra.sp = original_sp

        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"], "x=2")

    def test_divmod_tool_marks_answer_present(self) -> None:
        result = divmod_tool("159 divided by 14")

        self.assertTrue(result["solved"])
        self.assertTrue(result["answer"].startswith("q="))


if __name__ == "__main__":
    unittest.main()
