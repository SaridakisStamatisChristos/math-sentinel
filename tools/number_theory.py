
from __future__ import annotations

import math
import re
from typing import Any, Dict, List


def gcd_tool(arg: str, state: Any = None) -> Dict[str, Any]:
    nums = [int(x) for x in re.findall(r"-?\d+", arg)]
    if len(nums) < 2:
        return {"ok": False, "result": "need two ints"}
    rendered = str(math.gcd(nums[0], nums[1]))
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}


def lcm_tool(arg: str, state: Any = None) -> Dict[str, Any]:
    nums = [int(x) for x in re.findall(r"-?\d+", arg)]
    if len(nums) < 2:
        return {"ok": False, "result": "need two ints"}
    g = math.gcd(nums[0], nums[1])
    out = abs(nums[0] * nums[1]) // g if g else 0
    rendered = str(out)
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}


def primality(arg: str, state: Any = None) -> Dict[str, Any]:
    m = re.search(r"-?\d+", arg)
    if not m:
        return {"ok": False, "result": "need int"}
    n = int(m.group(0))
    if n < 2:
        return {"ok": True, "result": "composite", "solved": True, "answer": "composite", "goal_progress": 1.0}
    d = 2
    while d * d <= n:
        if n % d == 0:
            return {"ok": True, "result": "composite", "solved": True, "answer": "composite", "goal_progress": 1.0}
        d += 1
    return {"ok": True, "result": "prime", "solved": True, "answer": "prime", "goal_progress": 1.0}


def factorize(arg: str, state: Any = None) -> Dict[str, Any]:
    m = re.search(r"-?\d+", arg)
    if not m:
        return {"ok": False, "result": "need int"}
    n = abs(int(m.group(0)))
    if n < 2:
        rendered = str(n)
        return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}
    factors: List[int] = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    rendered = "*".join(str(x) for x in factors)
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}


def modular_reduce(arg: str, state: Any = None) -> Dict[str, Any]:
    nums = [int(x) for x in re.findall(r"-?\d+", arg)]
    if len(nums) < 2 or nums[1] == 0:
        return {"ok": False, "result": "need n,m"}
    rendered = str(nums[0] % nums[1])
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}


def gcd_lcm(arg: str, state: Any = None) -> Dict[str, Any]:
    nums = [int(x) for x in re.findall(r"-?\d+", arg)]
    if len(nums) < 2:
        return {"ok": False, "result": "need two ints"}
    g = math.gcd(nums[0], nums[1])
    l = abs(nums[0] * nums[1]) // g if g else 0
    rendered = f"gcd={g}, lcm={l}"
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}
