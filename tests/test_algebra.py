from __future__ import annotations

import pytest

from tools.algebra import solve_linear_step


def test_solve_linear_step_handles_plus_minus_constant() -> None:
    # Exercise the regex fallback by forcing sympy off
    import tools.algebra as algebra

    original_sp = algebra.sp
    algebra.sp = None
    try:
        result = solve_linear_step("2x + -4 = 0")
    finally:
        algebra.sp = original_sp

    assert result["ok"] is True
    assert result["answer"] == "x=2"
