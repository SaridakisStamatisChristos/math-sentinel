from __future__ import annotations

from typing import Any

from .beam import beam_search
from .mcts import mcts_search


def resolve_search_mode(initial_state: Any, score_config: dict[str, Any] | None = None) -> str:
    score_config = score_config or {}
    mode = str(score_config.get("mode", "auto")).strip().lower()
    if mode in {"beam", "mcts"}:
        return mode
    domain = str(getattr(initial_state, "domain", "")).lower()
    if "plan" in domain:
        return "mcts"
    return "beam"


def run_search(*, initial_state: Any, score_config: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    mode = resolve_search_mode(initial_state, score_config)
    if mode == "mcts":
        return mcts_search(initial_state=initial_state, score_config=score_config, **kwargs)
    return beam_search(initial_state=initial_state, score_config=score_config, **kwargs)
