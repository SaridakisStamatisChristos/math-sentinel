from __future__ import annotations

from typing import Any, Callable, Tuple

from memory.hard_cases import HardCaseStore
from memory.lemma_store import LemmaStore
from memory.tactic_stats import TacticStats


def load_search_memory(cfg: dict[str, Any]) -> Tuple[LemmaStore, HardCaseStore, TacticStats]:
    lemma_store = LemmaStore()
    hard_cases = HardCaseStore(capacity=int(cfg["memory"]["hard_case_capacity"]))
    tactic_stats = TacticStats()
    lemma_store.load(cfg["memory"]["lemma_store_path"])
    hard_cases.load(cfg["memory"]["hard_cases_path"])
    tactic_stats.load(cfg["memory"]["tactic_stats_path"])
    return lemma_store, hard_cases, tactic_stats


def build_prompt_builder(
    reasoning_domain: Any,
    *,
    lemma_store: LemmaStore | None = None,
    hard_case_store: HardCaseStore | None = None,
    tactic_stats: TacticStats | None = None,
    retrieval_mode: str = "hybrid",
    embedding_model: str = "hashing",
    event_logger: Callable[..., Any] | None = None,
) -> Callable[[Any], str]:
    def _builder(state: Any) -> str:
        prompt = reasoning_domain.build_search_prompt(
            state,
            lemma_store=lemma_store,
            hard_case_store=hard_case_store,
            tactic_stats=tactic_stats,
            retrieval_mode=retrieval_mode,
            embedding_model=embedding_model,
            event_logger=event_logger,
        )
        return prompt

    return _builder


def build_action_bias_fn(tactic_stats: TacticStats | None) -> Callable[[Any, Any], float]:
    def _bias(state: Any, action: Any) -> float:
        if action is None:
            return 0.5
        base = tactic_stats.bias(state.domain, action.type.value) if tactic_stats is not None else 0.5
        retrieval_context = getattr(state, "metadata", {}).get("_retrieval_context", {})
        tool_priors = retrieval_context.get("tool_priors", {}) if isinstance(retrieval_context, dict) else {}
        failure_avoidance = retrieval_context.get("failure_avoidance", []) if isinstance(retrieval_context, dict) else []
        if getattr(action, "tool", ""):
            tool_bias = float(tool_priors.get(str(action.tool).lower(), tool_priors.get(str(action.tool), 0.0)))
            if tool_bias > 0.0:
                base = min(0.99, base + 0.35 * tool_bias)
        action_text = f"{getattr(action, 'tool', '')} {getattr(action, 'content', '')}".lower()
        if any(fragment.lower() in action_text for fragment in failure_avoidance if isinstance(fragment, str)):
            base = max(0.01, base - 0.25)
        return base

    return _bias
