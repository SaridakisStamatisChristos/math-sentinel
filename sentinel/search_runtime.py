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
) -> Callable[[Any], str]:
    def _builder(state: Any) -> str:
        return reasoning_domain.build_search_prompt(
            state,
            lemma_store=lemma_store,
            hard_case_store=hard_case_store,
            tactic_stats=tactic_stats,
        )

    return _builder


def build_action_bias_fn(tactic_stats: TacticStats | None) -> Callable[[Any, Any], float]:
    def _bias(state: Any, action: Any) -> float:
        if tactic_stats is None or action is None:
            return 0.5
        return tactic_stats.bias(state.domain, action.type.value)

    return _bias
