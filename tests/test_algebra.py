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


def test_divmod_tool_marks_answer_present() -> None:
    from tools.arithmetic import divmod_tool

    res = divmod_tool("159 divided by 14")
    assert res["solved"] is True
    assert res["answer"].startswith("q=")
