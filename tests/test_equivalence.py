from __future__ import annotations

import unittest

from proof.equivalence import equivalent


class EquivalenceTests(unittest.TestCase):
    def test_fraction_equivalence_accepts_reduced_forms(self) -> None:
        self.assertTrue(equivalent("fractions", "2/4", "1/2", {"family": "fractions"}))

    def test_divmod_equivalence_ignores_field_order(self) -> None:
        self.assertTrue(equivalent("divmod", "r=1, q=3", "q=3, r=1", {"family": "divmod"}))

    def test_symbolic_derivative_equivalence_uses_algebra(self) -> None:
        self.assertTrue(equivalent("derivative", "2*x + 2*x", "4*x", {"family": "derivative"}))

    def test_integral_equivalence_allows_constant_offset(self) -> None:
        self.assertTrue(equivalent("integral", "x**2 + 7", "x**2 + 3", {"family": "integral"}))

    def test_linear_equation_equivalence_rejects_wrong_answer(self) -> None:
        self.assertFalse(equivalent("linear_equation", "x=5", "x=4", {"family": "linear_equation"}))

    def test_parity_proof_equivalence_accepts_semantic_variant(self) -> None:
        candidate = "Because 14 = 2*7, it is divisible by 2 and therefore even."
        expected = "Because 2n = 2*k for k=n, it is divisible by 2 and therefore even."
        self.assertTrue(equivalent("parity_proof", candidate, expected, {"family": "parity_proof"}))


if __name__ == "__main__":
    unittest.main()