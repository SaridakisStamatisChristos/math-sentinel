
from __future__ import annotations

from typing import Any, Dict


def square(arg: str, state: Any = None) -> Dict[str, Any]:
    try:
        n = int(arg.strip())
        return {"ok": True, "result": str(n * n)}
    except Exception:
        return {"ok": False, "result": "need integer"}


def parity_claim(arg: str, state: Any = None) -> Dict[str, Any]:
    text = arg.strip().lower()
    if "even" in text and "2*" in text:
        return {"ok": True, "result": "plausible evenness witness"}
    return {"ok": False, "result": "unsupported parity claim"}


def register(registry: Any) -> None:
    registry.register("square", square)
    registry.register("parity_claim", parity_claim)
