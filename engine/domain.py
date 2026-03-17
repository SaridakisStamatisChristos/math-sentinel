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

    def render_human_trace(self, state: ReasoningState) -> str: ...

    def create_executor(self) -> Any: ...

    def maybe_derive_lemma(self, task: TaskT) -> Optional[Any]: ...
