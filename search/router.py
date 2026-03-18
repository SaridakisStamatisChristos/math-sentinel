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
    def _annotate_search_audit(final_state: Any, explored_nodes: list[SearchNode], *, resolved_mode: str) -> Any:
        metadata = getattr(final_state, "metadata", {})
        if not isinstance(metadata, dict):
            return final_state
        fallback_count = sum(1 for node in explored_nodes if float((getattr(node, "local_scores", {}) or {}).get("fallback_repair_used", 0.0)) > 0.0)
        guided_count = sum(1 for node in explored_nodes if float((getattr(node, "local_scores", {}) or {}).get("guided_rollout_used", 0.0)) > 0.0)
        fallback_chain_count = sum(1 for node in explored_nodes if float((getattr(node, "local_scores", {}) or {}).get("fallback_chain_used", 0.0)) > 0.0)
        metadata["search_audit"] = {
            "mode": resolved_mode,
            "fallback_repair_used": fallback_count > 0,
            "fallback_repair_attempts": fallback_count,
            "guided_rollout_used": guided_count > 0,
            "guided_rollout_steps": guided_count,
            "fallback_chain_used": fallback_chain_count > 0,
            "fallback_chain_steps": fallback_chain_count,
        }
        return final_state

    mode = resolve_search_mode(initial_state, score_config)
    if mode == "mcts":
        final_state, explored = mcts_search(initial_state=initial_state, score_config=score_config, **kwargs)
    else:
        final_state, explored = beam_search(initial_state=initial_state, score_config=score_config, **kwargs)

    score_config = score_config or {}
    final_status = getattr(final_state, "status", None)
    if final_status != "solved" and final_status is not None and bool(score_config.get("deterministic_fallback_chain", False)):
        executor = kwargs.get("executor")
        fallback_repairs_fn = kwargs.get("fallback_repairs_fn")
        event_logger = kwargs.get("event_logger")
        if executor is not None and callable(fallback_repairs_fn):
            chain_state = initial_state.clone()
            chain_explored: list[SearchNode] = [SearchNode(state=chain_state, cumulative_score=0.0, depth=0)]
            chain_depth = int(score_config.get("fallback_chain_depth", max(int(score_config.get("max_depth", kwargs.get("max_depth", 4))), 8)))
            for depth in range(1, chain_depth + 1):
                repairs = fallback_repairs_fn(chain_state)
                if not repairs:
                    break
                action = repairs[0]
                chain_state, local = executor.apply(chain_state, action)
                chain_local_scores = {
                    **{key: float(value) for key, value in local.items() if isinstance(value, (int, float))},
                    "fallback_repair_used": 1.0,
                    "fallback_chain_used": 1.0,
                }
                chain_explored.append(
                    SearchNode(
                        state=chain_state,
                        cumulative_score=float(depth),
                        local_scores=chain_local_scores,
                        parent=chain_explored[-1],
                        action=action,
                        depth=depth,
                    )
                )
                if event_logger is not None:
                    event_logger("fallback_chain_step", domain=initial_state.domain, depth=depth, mode=mode, tool=action.tool)
                if chain_state.status == "solved":
                    if event_logger is not None:
                        event_logger("fallback_chain_used", domain=initial_state.domain, depth=depth, mode=mode)
                    annotated = _annotate_search_audit(chain_state, explored + chain_explored[1:], resolved_mode=mode)
                    return annotated, explored + chain_explored[1:]
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
                rollout_local_scores = {
                    **{key: float(value) for key, value in local.items() if isinstance(value, (int, float))},
                    "guided_rollout_used": 1.0,
                    "fallback_repair_used": 1.0,
                }
                rollout_explored.append(
                    SearchNode(
                        state=rollout_state,
                        cumulative_score=float(depth),
                        local_scores=rollout_local_scores,
                        parent=rollout_explored[-1],
                        action=action,
                        depth=depth,
                    )
                )
                if event_logger is not None:
                    event_logger("guided_rollout_step", domain=initial_state.domain, depth=depth, mode=mode, tool=action.tool)
                if rollout_state.status == "solved":
                    if event_logger is not None:
                        event_logger("guided_rollout_used", domain=initial_state.domain, depth=depth, mode=mode)
                    annotated = _annotate_search_audit(rollout_state, explored + rollout_explored[1:], resolved_mode=mode)
                    return annotated, explored + rollout_explored[1:]
            if rollout_state.status == "solved":
                annotated = _annotate_search_audit(rollout_state, explored + rollout_explored[1:], resolved_mode=mode)
                return annotated, explored + rollout_explored[1:]
    annotated = _annotate_search_audit(final_state, explored, resolved_mode=mode)
    return annotated, explored
