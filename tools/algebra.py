
from __future__ import annotations

import re
from typing import Any, Dict

try:
    import sympy as sp
except Exception:
    sp = None


def solve_linear_step(arg: str, state: Any = None) -> Dict[str, Any]:
    text = arg.strip().replace(" ", "")
    if "=" not in text:
        return {"ok": False, "result": "missing ="}
    left, right = text.split("=", 1)
    if sp is not None:
        x = sp.Symbol("x")
        try:
            sol = sp.solve(sp.Eq(sp.sympify(left, locals={"x": x}), sp.sympify(right, locals={"x": x})), x)
            if len(sol) == 1:
                return {"ok": True, "result": f"x={sol[0]}", "solved": True, "answer": f"x={sol[0]}", "goal_progress": 1.0}
        except Exception:
            pass
    m = re.match(r"([+-]?\d*)x([+-]\d+)?", left)
    if not m:
        return {"ok": False, "result": "unsupported linear form"}
    a_txt = m.group(1)
    b_txt = m.group(2) or "+0"
    a = int(a_txt or "1") if a_txt not in {"", "+", "-"} else (1 if a_txt != "-" else -1)
    b = int(b_txt)
    c = int(right)
    if a == 0:
        return {"ok": False, "result": "a=0"}
    x_val = (c - b) / a
    if int(x_val) == x_val:
        x_val = int(x_val)
    return {"ok": True, "result": f"x={x_val}", "solved": True, "answer": f"x={x_val}", "goal_progress": 1.0}


def simplify_polynomial(arg: str, state: Any = None) -> Dict[str, Any]:
    expr = arg.strip()
    if sp is not None:
        x = sp.Symbol("x")
        try:
            out = sp.expand(sp.sympify(expr, locals={"x": x}))
            rendered = str(out)
            return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}
        except Exception:
            pass
    expr = expr.replace(" ", "").replace("+-", "-")
    return {"ok": True, "result": expr, "solved": True, "answer": expr, "goal_progress": 1.0}


def expand_or_factor(arg: str, state: Any = None) -> Dict[str, Any]:
    expr = arg.strip()
    if sp is not None:
        x = sp.Symbol("x")
        try:
            sym = sp.sympify(expr, locals={"x": x})
            fact = sp.factor(sym)
            return {"ok": True, "result": str(fact)}
        except Exception:
            pass
    return {"ok": True, "result": expr}


def normalize_expression(arg: str, state: Any = None) -> Dict[str, Any]:
    return {"ok": True, "result": arg.strip().replace(" ", "").replace("+-", "-")}
