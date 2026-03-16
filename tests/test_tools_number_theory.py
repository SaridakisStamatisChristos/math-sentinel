from __future__ import annotations

from tools.number_theory import gcd_tool, lcm_tool, gcd_lcm


def test_gcd_lcm_tools_return_solved_with_answer() -> None:
    g = gcd_tool("gcd of 12 and 18")
    l = lcm_tool("lcm(12,18)")
    both = gcd_lcm("Compute gcd and lcm of 12 and 18")

    assert g["solved"] and g["answer"] == "6"
    assert l["solved"] and l["answer"] == "36"
    assert both["solved"] and both["answer"] == "gcd=6, lcm=36"
