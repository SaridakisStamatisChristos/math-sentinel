
from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Dict


def reduce_fraction(arg: str, state: Any = None) -> Dict[str, Any]:
    m = re.search(r"(-?\d+)\s*/\s*(-?\d+)", arg)
    if not m:
        return {"ok": False, "result": "need a/b"}
    a, b = int(m.group(1)), int(m.group(2))
    if b == 0:
        return {"ok": False, "result": "division by zero"}
    f = Fraction(a, b)
    rendered = f"{f.numerator}/{f.denominator}"
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}


def compare_fractions(arg: str, state: Any = None) -> Dict[str, Any]:
    parts = re.findall(r"-?\d+\s*/\s*-?\d+", arg)
    if len(parts) < 2:
        return {"ok": False, "result": "need two fractions"}
    a, b = Fraction(parts[0]), Fraction(parts[1])
    if a < b:
        sign = "<"
    elif a > b:
        sign = ">"
    else:
        sign = "="
    return {"ok": True, "result": sign}


def common_denominator(arg: str, state: Any = None) -> Dict[str, Any]:
    parts = re.findall(r"-?\d+\s*/\s*-?\d+", arg)
    if len(parts) < 2:
        return {"ok": False, "result": "need two fractions"}
    fracs = [Fraction(p) for p in parts[:2]]
    denom = fracs[0].denominator * fracs[1].denominator
    return {"ok": True, "result": str(denom)}
