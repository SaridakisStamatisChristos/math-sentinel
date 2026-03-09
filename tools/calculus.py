
from __future__ import annotations

from typing import Any, Dict

try:
    import sympy as sp
except Exception:
    sp = None


def derivative(arg: str, state: Any = None) -> Dict[str, Any]:
    expr = arg.strip()
    if sp is None:
        return {"ok": False, "result": "sympy unavailable"}
    x = sp.Symbol("x")
    try:
        out = sp.diff(sp.sympify(expr, locals={"x": x}), x)
        rendered = str(sp.expand(out))
        return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}
    except Exception as exc:
        return {"ok": False, "result": f"derivative error: {exc}"}


def antiderivative(arg: str, state: Any = None) -> Dict[str, Any]:
    expr = arg.strip()
    if sp is None:
        return {"ok": False, "result": "sympy unavailable"}
    x = sp.Symbol("x")
    try:
        out = sp.integrate(sp.sympify(expr, locals={"x": x}), x)
        rendered = str(sp.expand(out)) + " + C"
        return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}
    except Exception as exc:
        return {"ok": False, "result": f"integral error: {exc}"}


def simplify_calculus_form(arg: str, state: Any = None) -> Dict[str, Any]:
    if sp is None:
        return {"ok": True, "result": arg.strip()}
    x = sp.Symbol("x")
    try:
        out = sp.simplify(sp.sympify(arg, locals={"x": x}))
        return {"ok": True, "result": str(out)}
    except Exception:
        return {"ok": True, "result": arg.strip()}
