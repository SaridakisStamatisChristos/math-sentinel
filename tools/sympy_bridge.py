
from __future__ import annotations

from typing import Any, Dict

try:
    import sympy as sp
except Exception:
    sp = None


def available() -> bool:
    return sp is not None


def simplify_expr(arg: str, state: Any = None) -> Dict[str, Any]:
    if sp is None:
        return {"ok": False, "result": "sympy unavailable"}
    x = sp.Symbol("x")
    try:
        out = sp.simplify(sp.sympify(arg, locals={"x": x}))
        return {"ok": True, "result": str(out)}
    except Exception as exc:
        return {"ok": False, "result": f"sympy error: {exc}"}


def equivalent_expr(arg: str, state: Any = None) -> Dict[str, Any]:
    if sp is None:
        return {"ok": False, "result": "sympy unavailable"}
    parts = [p.strip() for p in arg.split("==")]
    if len(parts) != 2:
        return {"ok": False, "result": "use a==b"}
    x = sp.Symbol("x")
    try:
        ok = sp.simplify(sp.sympify(parts[0], locals={"x": x}) - sp.sympify(parts[1], locals={"x": x})) == 0
        return {"ok": True, "result": "true" if ok else "false"}
    except Exception as exc:
        return {"ok": False, "result": f"sympy error: {exc}"}
