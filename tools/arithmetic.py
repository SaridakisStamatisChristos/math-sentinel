
from __future__ import annotations

import re
from typing import Dict, Any


def add(arg: str, state: Any = None) -> Dict[str, Any]:
    nums = [int(x) for x in re.findall(r"-?\d+", arg)]
    if len(nums) < 2:
        return {"ok": False, "result": "need two ints"}
    out = str(sum(nums))
    return {"ok": True, "result": out, "solved": True, "answer": out, "goal_progress": 1.0}


def subtract(arg: str, state: Any = None) -> Dict[str, Any]:
    nums = [int(x) for x in re.findall(r"-?\d+", arg)]
    if len(nums) < 2:
        return {"ok": False, "result": "need two ints"}
    out = str(nums[0] - nums[1])
    return {"ok": True, "result": out, "solved": True, "answer": out, "goal_progress": 1.0}


def multiply(arg: str, state: Any = None) -> Dict[str, Any]:
    nums = [int(x) for x in re.findall(r"-?\d+", arg)]
    if len(nums) < 2:
        return {"ok": False, "result": "need two ints"}
    out = 1
    for n in nums:
        out *= n
    rendered = str(out)
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}


def divmod_tool(arg: str, state: Any = None) -> Dict[str, Any]:
    nums = [int(x) for x in re.findall(r"-?\d+", arg)]
    if len(nums) < 2 or nums[1] == 0:
        return {"ok": False, "result": "bad args"}
    q, r = divmod(nums[0], nums[1])
    rendered = f"q={q}, r={r}"
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}
