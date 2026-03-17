
from __future__ import annotations

import json
import re
from typing import List, Tuple

from .actions import Action, ActionType


ACTION_RE = re.compile(
    r'<action\s+type="(?P<atype>[A-Z_]+)"(?:\s+tool="(?P<tool>[^"]+)")?(?:\s+name="(?P<name>[^"]+)")?>(?P<content>.*?)</action>',
    re.DOTALL,
)
CANONICAL_ACTION_RE = re.compile(r"^ACTION\s+(?P<payload>\{.*\})\s*$")


def _parse_canonical_actions(text: str) -> Tuple[List[Action], float]:
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("ACTION ")]
    if not lines:
        return ([], 0.0)

    actions: List[Action] = []
    valid_count = 0
    for line in lines:
        m = CANONICAL_ACTION_RE.match(line)
        if not m:
            continue
        try:
            payload = json.loads(m.group("payload"))
            atype = str(payload.get("type", "")).strip().upper()
            content = str(payload.get("content", ""))
            tool = str(payload.get("tool", "")).strip()
            name = str(payload.get("name", "")).strip()
            raw_payload = payload.get("payload", {})
            extra = raw_payload if isinstance(raw_payload, dict) else {}
            action = Action(type=ActionType(atype), content=content, tool=tool, name=name, payload=extra)
            if action.validate():
                valid_count += 1
                actions.append(action)
        except Exception:
            continue
    return actions, (valid_count / max(1, len(lines)))


def parse_actions(text: str) -> Tuple[List[Action], float]:
    canonical_actions, canonical_confidence = _parse_canonical_actions(text)
    if canonical_actions:
        return canonical_actions, canonical_confidence

    actions: List[Action] = []
    matches = list(ACTION_RE.finditer(text))
    if not matches:
        content = text.strip()
        if "<answer>" in content and "</answer>" in content:
            answer = content.split("<answer>", 1)[1].split("</answer>", 1)[0]
            action = Action(type=ActionType.ANSWER, content=answer.strip())
            return ([action], 0.55 if action.validate() else 0.15)
        return ([], 0.0)

    valid_count = 0
    for m in matches:
        atype = m.group("atype")
        tool = (m.group("tool") or "").strip()
        name = (m.group("name") or "").strip()
        content = (m.group("content") or "").strip()
        try:
            a = Action(type=ActionType(atype), content=content, tool=tool, name=name)
            if a.validate():
                valid_count += 1
                actions.append(a)
        except Exception:
            continue
    confidence = valid_count / max(1, len(matches))
    return actions, confidence
