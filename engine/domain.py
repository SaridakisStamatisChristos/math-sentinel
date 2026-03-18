from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, TypeVar

import torch

from .state import ReasoningState


TaskT = TypeVar("TaskT")


class ReasoningDomain(Protocol[TaskT]):
    name: str

    def sample_task(self, domains: List[str]) -> TaskT: ...

    def make_state(self, task: TaskT) -> ReasoningState: ...

    def manual_task(self, domain: str, prompt: str, answer: str = "") -> TaskT: ...

    def build_training_example(self, task: TaskT) -> str: ...

    def build_verifier_examples(self, task: TaskT) -> Tuple[str, torch.Tensor, str, torch.Tensor]: ...

    def build_verifier_targets(
        self,
        task: TaskT,
        state: ReasoningState,
        local_scores: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor: ...

    def evaluate_answer(self, task: TaskT, candidate: str) -> bool: ...

    def parse_actions(self, text: str) -> Tuple[List[Any], float]: ...

    def fallback_repairs(self, state: ReasoningState) -> List[Any]: ...

    def action_schema(self, state: ReasoningState) -> Dict[str, Any]: ...

    def allowed_action_types(self, state: ReasoningState) -> List[str]: ...

    def allowed_tools(self, state: ReasoningState, action_type: str) -> List[str]: ...

    def candidate_bindings(self, state: ReasoningState, action_type: str, tool: str = "") -> List[Dict[str, str]]: ...

    def action_format_instructions(self) -> str: ...

    def build_search_prompt(
        self,
        state: ReasoningState,
        *,
        lemma_store: Any | None = None,
        hard_case_store: Any | None = None,
        tactic_stats: Any | None = None,
        retrieval_mode: str = "hybrid",
        embedding_model: str = "hashing",
        event_logger: Any | None = None,
    ) -> str: ...

    def state_signature(self, state: ReasoningState) -> str: ...

    def render_human_trace(self, state: ReasoningState) -> str: ...

    def create_executor(self) -> Any: ...

    def maybe_derive_lemma(self, task: TaskT) -> Optional[Any]: ...

    def benchmark_tasks(self) -> List[TaskT]: ...
