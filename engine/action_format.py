from __future__ import annotations

import json
from typing import Iterable

from .actions import Action


def render_canonical_action(action: Action) -> str:
    payload = {
        "type": action.type.value,
        "content": action.content,
    }
    if action.tool:
        payload["tool"] = action.tool
    if action.name:
        payload["name"] = action.name
    if action.payload:
        payload["payload"] = action.payload
    return "ACTION " + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def render_canonical_actions(actions: Iterable[Action]) -> str:
    return "\n".join(render_canonical_action(action) for action in actions)
