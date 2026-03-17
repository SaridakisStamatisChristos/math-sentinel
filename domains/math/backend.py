from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from curriculum.generators import GeneratedTask, sample_task
from curriculum.oracle import evaluate_answer
from curriculum.trajectory_builder import build_gold_trace
from engine.state import ReasoningState
from proof.executor import ProofExecutor
from proof.lemmas import (
    Lemma,
    derive_arithmetic_lemma,
    derive_calculus_lemma,
    derive_divmod_lemma,
    derive_factorization_lemma,
    derive_fractions_lemma,
    derive_gcd_lcm_lemma,
    derive_linear_lemma,
    derive_logic_lemma,
    derive_modular_lemma,
    derive_parity_proof_lemma,
    derive_polynomial_lemma,
    derive_primality_lemma,
)
from proof.parser import parse_actions
from proof.state import ProofState
from proof.traces import render_human_trace
from search.repair import fallback_repairs
from tools.registry import ToolRegistry


class MathReasoningDomain:
    name = "math"
    default_curriculum_config = "config/curriculum.yaml"

    def __init__(self, checker_plugin: str = "") -> None:
        self.checker_plugin = checker_plugin

    def sample_task(self, domains: List[str]) -> GeneratedTask:
        return sample_task(domains)

    def make_state(self, task: GeneratedTask) -> ProofState:
        return ProofState(
            task_id=task.task_id,
            domain=task.domain,
            problem_text=task.prompt,
            goal=task.goal,
            expected_answer=task.answer,
            metadata=task.meta,
        )

    def manual_task(self, domain: str, prompt: str, answer: str = "") -> GeneratedTask:
        return GeneratedTask(
            task_id="manual_0001",
            domain=domain,
            prompt=prompt,
            answer=answer,
            goal="Solve the problem",
            meta={"family": domain},
        )

    def build_training_example(self, task: GeneratedTask) -> str:
        state = self.make_state(task)
        return state.serialize() + "\n" + build_gold_trace(task)

    def build_verifier_examples(self, task: GeneratedTask) -> tuple[str, torch.Tensor, str, torch.Tensor]:
        pos = self.make_state(task)
        pos.final_answer = task.answer
        pos.status = "solved"
        pos.derived_facts.append(task.answer)
        pos.action_history.append({"type": "ANSWER", "content": task.answer})
        pos.tool_history.append({"tool": "oracle", "result": {"ok": True, "answer": task.answer}})

        neg = self.make_state(task)
        neg.derived_facts.append("search_not_started")

        pos_t = self.build_verifier_targets(task, pos)
        neg_t = self.build_verifier_targets(task, neg, local_scores={"valid_step": 0.35, "goal_progress": 0.0, "risk_score": 0.8})
        return pos.serialize(), pos_t, neg.serialize(), neg_t

    def build_verifier_targets(
        self,
        task: GeneratedTask,
        state: ReasoningState,
        local_scores: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        local_scores = local_scores or {}
        has_answer = bool(state.final_answer.strip())
        correct = has_answer and self.evaluate_answer(task, state.final_answer)
        solved = state.status == "solved"

        valid_step = float(local_scores.get("valid_step", 1.0 if solved or has_answer else 0.6))
        explicit_progress = float(local_scores.get("goal_progress", 0.0))
        structural_progress = min(
            1.0,
            0.12 * len(state.derived_facts)
            + 0.08 * len(state.tool_history)
            + 0.06 * len(state.action_history)
            + 0.08 * len(state.lemma_refs)
            + 0.04 * len(state.subgoals),
        )
        goal_progress = max(explicit_progress, structural_progress)
        if correct:
            goal_progress = max(goal_progress, 0.95)
        elif has_answer:
            goal_progress = min(0.75, max(goal_progress, 0.3))

        proof_completion = 1.0 if correct and solved else (0.25 if has_answer else min(0.2, goal_progress * 0.5))
        risk = float(local_scores.get("risk_score", 0.05 if correct else (0.85 if has_answer else 0.45)))
        branch_priority = max(0.05, min(0.98, 0.55 * goal_progress + 0.25 * valid_step + 0.20 * proof_completion))

        values = [
            max(0.02, min(0.99, valid_step)),
            max(0.0, min(0.99, goal_progress)),
            max(0.0, min(0.99, proof_completion)),
            max(0.01, min(0.99, risk)),
            branch_priority,
        ]
        return torch.tensor(values, dtype=torch.float32)

    def evaluate_answer(self, task: GeneratedTask, candidate: str) -> bool:
        return evaluate_answer(task, candidate)

    def parse_actions(self, text: str) -> tuple[List[Any], float]:
        return parse_actions(text)

    def fallback_repairs(self, state: ProofState) -> List[Any]:
        return fallback_repairs(state)

    def render_human_trace(self, state: ProofState) -> str:
        return render_human_trace(state)

    def create_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        if self.checker_plugin:
            registry.load_plugin(self.checker_plugin)
        return registry

    def create_executor(self) -> ProofExecutor:
        return ProofExecutor(self.create_registry())

    def maybe_derive_lemma(self, task: GeneratedTask) -> Optional[Lemma]:
        if task.domain == "linear_equation":
            return derive_linear_lemma(task.prompt)
        if task.domain == "polynomial_simplify":
            return derive_polynomial_lemma(task.prompt)
        if task.domain in {"derivative", "integral"}:
            return derive_calculus_lemma(task.prompt)
        if task.domain == "logic":
            return derive_logic_lemma(task.prompt)
        if task.domain == "arithmetic":
            return derive_arithmetic_lemma(task.prompt)
        if task.domain == "fractions":
            return derive_fractions_lemma(task.prompt)
        if task.domain == "divmod":
            return derive_divmod_lemma(task.prompt)
        if task.domain == "gcd_lcm":
            return derive_gcd_lcm_lemma(task.prompt)
        if task.domain == "modular":
            return derive_modular_lemma(task.prompt)
        if task.domain == "primality":
            return derive_primality_lemma(task.prompt)
        if task.domain == "factorization":
            return derive_factorization_lemma(task.prompt)
        if task.domain == "parity_proof":
            return derive_parity_proof_lemma(task.prompt)
        return None
