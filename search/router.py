from __future__ import annotations

from typing import Any

from .beam import beam_search
from .mcts import mcts_search
from .nodes import SearchNode


def resolve_search_mode(initial_state: Any, score_config: dict[str, Any] | None = None) -> str:
    score_config = score_config or {}
    mode = str(score_config.get("mode", "auto")).strip().lower()
    if mode in {"beam", "mcts"}:
        return mode
    domain = str(getattr(initial_state, "domain", "")).lower()
    metadata = getattr(initial_state, "metadata", {}) or {}
    family = str(metadata.get("family", "")).lower()
    combined = f"{domain} {family}"
    if any(token in combined for token in ["plan", "gaia", "swebench", "repo_patch"]):
        return "mcts"
    return "beam"


def run_search(*, initial_state: Any, score_config: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    mode = resolve_search_mode(initial_state, score_config)
    if mode == "mcts":
        final_state, explored = mcts_search(initial_state=initial_state, score_config=score_config, **kwargs)
    else:
        final_state, explored = beam_search(initial_state=initial_state, score_config=score_config, **kwargs)

    score_config = score_config or {}
    final_status = getattr(final_state, "status", None)
    if final_status != "solved" and final_status is not None and bool(score_config.get("guided_fallback_rollout", True)):
        executor = kwargs.get("executor")
        fallback_repairs_fn = kwargs.get("fallback_repairs_fn")
        event_logger = kwargs.get("event_logger")
        if executor is not None and callable(fallback_repairs_fn):
            rollout_state = initial_state.clone()
            rollout_explored: list[SearchNode] = [SearchNode(state=rollout_state, cumulative_score=0.0, depth=0)]
            guided_rollout_depth = int(
                score_config.get(
                    "guided_rollout_depth",
                    max(int(score_config.get("max_depth", kwargs.get("max_depth", 4))), int(kwargs.get("max_depth", 4)), 6),
                )
            )
            for depth in range(1, guided_rollout_depth + 1):
                repairs = fallback_repairs_fn(rollout_state)
                if not repairs:
                    break
                action = repairs[0]
                rollout_state, local = executor.apply(rollout_state, action)
                rollout_explored.append(
                    SearchNode(
                        state=rollout_state,
                        cumulative_score=float(depth),
                        local_scores={key: float(value) for key, value in local.items() if isinstance(value, (int, float))},
                        parent=rollout_explored[-1],
                        action=action,
                        depth=depth,
                    )
                )
                if rollout_state.status == "solved":
                    if event_logger is not None:
                        event_logger("guided_rollout_used", domain=initial_state.domain, depth=depth, mode=mode)
                    return rollout_state, explored + rollout_explored[1:]
            if rollout_state.status == "solved":
                return rollout_state, explored + rollout_explored[1:]
    return final_state, explored
