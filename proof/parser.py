
from __future__ import annotations

import re
from typing import List, Tuple

from .actions import Action, ActionType


ACTION_RE = re.compile(
    r'<action\s+type="(?P<atype>[A-Z_]+)"(?:\s+tool="(?P<tool>[^"]+)")?(?:\s+name="(?P<name>[^"]+)")?>(?P<content>.*?)</action>',
    re.DOTALL,
)


def parse_actions(text: str) -> Tuple[List[Action], float]:
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
