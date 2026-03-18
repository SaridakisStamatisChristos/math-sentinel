from __future__ import annotations

from typing import Any, Dict


def normalize_tool_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(raw.get("result_payload", raw.get("payload", {})) or {})
    result_text = str(raw.get("result_text", raw.get("result", "")) or "")
    answer = raw.get("answer")
    if answer is None and raw.get("solved"):
        answer = result_text
    normalized = {
        "ok": bool(raw.get("ok", False)),
        "result_text": result_text,
        "result_payload": payload,
        "goal_progress": float(raw.get("goal_progress", 0.0)),
        "solved": bool(raw.get("solved", False)),
        "answer": "" if answer is None else str(answer),
        "risk": float(raw.get("risk", 0.0)),
    }
    if "result" in raw and not normalized["result_text"]:
        normalized["result_text"] = str(raw["result"])
    return normalized
