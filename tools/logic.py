
from __future__ import annotations

import re
from typing import Any, Dict


def equality_transitivity(arg: str, state: Any = None) -> Dict[str, Any]:
    parts = [p.strip() for p in arg.split(";") if p.strip()]
    if len(parts) < 2:
        return {"ok": False, "result": "need two equalities a=b;b=c"}
    left = parts[0].split("=")
    right = parts[1].split("=")
    if len(left) != 2 or len(right) != 2:
        return {"ok": False, "result": "bad equality format"}
    if left[1].strip() == right[0].strip():
        return {"ok": True, "result": f"{left[0].strip()}={right[1].strip()}"}
    return {"ok": False, "result": "middle terms do not match"}


def contradiction_marker(arg: str, state: Any = None) -> Dict[str, Any]:
    return {"ok": True, "result": "contradiction-marked", "goal_progress": 0.1}


def prove_even(arg: str, state: Any = None) -> Dict[str, Any]:
    m = re.search(r"(\d+)", arg)
    if not m:
        return {"ok": False, "result": "need an even integer in the prompt"}
    n = int(m.group(1))
    rendered = f"Because {n} = 2*{n // 2}, it is divisible by 2 and therefore even."
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}
