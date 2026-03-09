from __future__ import annotations

import unittest

from tools.algebra import simplify_polynomial
from tools.calculus import antiderivative, derivative
from tools.logic import prove_even


class TerminalToolTests(unittest.TestCase):
    def test_symbolic_tools_mark_terminal_answers(self) -> None:
        deriv = derivative("x**2 + 3*x")
        integ = antiderivative("2*x")
        poly = simplify_polynomial("(2*x + 1) + (3*x)")

        self.assertTrue(deriv["solved"])
        self.assertTrue(integ["solved"])
        self.assertTrue(poly["solved"])

    def test_parity_tool_returns_even_proof(self) -> None:
        result = prove_even("Give a short proof sketch that 20 is even.")

        self.assertTrue(result["solved"])
        self.assertIn("even", result["answer"].lower())


if __name__ == "__main__":
    unittest.main()