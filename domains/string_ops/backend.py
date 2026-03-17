from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

import torch

from engine.actions import Action, ActionType
from engine.executor import StateExecutor
from engine.state import ReasoningState
from engine.task import ReasoningTask
from engine.traces import render_human_trace
from proof.parser import parse_actions


WORDS = [
    "amber",
    "banana",
    "cinder",
    "delta",
    "ember",
    "forest",
    "gloss",
    "harbor",
    "ivory",
    "jungle",
    "kernel",
    "lantern",
    "marble",
    "nectar",
    "orbit",
    "prairie",
    "quartz",
    "ripple",
    "signal",
    "thunder",
]


def _rid(prefix: str) -> str:
    return f"{prefix}_{random.randint(10**7, 10**8 - 1)}"


def _pick_words(count: int) -> list[str]:
    return random.sample(WORDS, count)


def gen_reverse_text() -> ReasoningTask:
    word = random.choice(WORDS)
    return ReasoningTask(
        task_id=_rid("strrev"),
        domain="reverse_text",
        prompt=f"Reverse the text: {word}",
        answer=word[::-1],
        goal="Return the reversed text exactly",
        meta={"family": "reverse_text", "text": word},
    )


def gen_uppercase_text() -> ReasoningTask:
    word = random.choice(WORDS)
    return ReasoningTask(
        task_id=_rid("strup"),
        domain="uppercase_text",
        prompt=f"Convert to uppercase: {word}",
        answer=word.upper(),
        goal="Return the uppercase form",
        meta={"family": "uppercase_text", "text": word},
    )


def gen_vowel_count() -> ReasoningTask:
    word = random.choice(WORDS)
    count = sum(ch in "aeiou" for ch in word.lower())
    return ReasoningTask(
        task_id=_rid("vowel"),
        domain="vowel_count",
        prompt=f"Count vowels in: {word}",
        answer=str(count),
        goal="Return the vowel count as an integer",
        meta={"family": "vowel_count", "text": word},
    )


def gen_sort_words() -> ReasoningTask:
    words = _pick_words(3)
    answer = " ".join(sorted(words))
    prompt_words = " ".join(words)
    return ReasoningTask(
        task_id=_rid("sort"),
        domain="sort_words",
        prompt=f"Sort words alphabetically: {prompt_words}",
        answer=answer,
        goal="Return the words in alphabetical order separated by spaces",
        meta={"family": "sort_words", "words": words},
    )


def gen_dedupe_words() -> ReasoningTask:
    base = _pick_words(3)
    words = [base[0], base[1], base[0], base[2], base[1]]
    answer = " ".join([base[0], base[1], base[2]])
    return ReasoningTask(
        task_id=_rid("dedupe"),
        domain="dedupe_words",
        prompt=f"Remove duplicate words preserving order: {' '.join(words)}",
        answer=answer,
        goal="Return the first occurrence of each word in order",
        meta={"family": "dedupe_words", "words": words},
    )


GENERATORS = {
    "reverse_text": gen_reverse_text,
    "uppercase_text": gen_uppercase_text,
    "vowel_count": gen_vowel_count,
    "sort_words": gen_sort_words,
    "dedupe_words": gen_dedupe_words,
}


def sample_task(domains: List[str]) -> ReasoningTask:
    domain = random.choice(domains)
    return GENERATORS[domain]()


def _content_after_colon(arg: str) -> str:
    return arg.split(":", 1)[-1].strip()


def reverse_text(arg: str, state: Any = None) -> Dict[str, Any]:
    text = _content_after_colon(arg)
    rendered = text[::-1]
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}


def uppercase_text(arg: str, state: Any = None) -> Dict[str, Any]:
    text = _content_after_colon(arg)
    rendered = text.upper()
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}


def vowel_count(arg: str, state: Any = None) -> Dict[str, Any]:
    text = _content_after_colon(arg)
    rendered = str(sum(ch in "aeiou" for ch in text.lower()))
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}


def sort_words(arg: str, state: Any = None) -> Dict[str, Any]:
    words = _content_after_colon(arg).split()
    rendered = " ".join(sorted(words))
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}


def dedupe_words(arg: str, state: Any = None) -> Dict[str, Any]:
    seen: set[str] = set()
    kept: list[str] = []
    for word in _content_after_colon(arg).split():
        if word not in seen:
            kept.append(word)
            seen.add(word)
    rendered = " ".join(kept)
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}


class StringToolRegistry:
    def __init__(self) -> None:
        self.tools = {
            "reverse_text": reverse_text,
            "uppercase_text": uppercase_text,
            "vowel_count": vowel_count,
            "sort_words": sort_words,
            "dedupe_words": dedupe_words,
        }

    def call(self, name: str, arg: str, state: Any = None) -> Dict[str, Any]:
        fn = self.tools.get(name)
        if fn is None:
            return {"ok": False, "result": f"unknown tool: {name}"}
        return fn(arg, state)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


class StringOpsReasoningDomain:
    name = "string_ops"
    default_curriculum_config = "config/string_ops_curriculum.yaml"

    def sample_task(self, domains: List[str]) -> ReasoningTask:
        return sample_task(domains)

    def make_state(self, task: ReasoningTask) -> ReasoningState:
        return ReasoningState(
            task_id=task.task_id,
            domain=task.domain,
            problem_text=task.prompt,
            goal=task.goal,
            expected_answer=task.answer,
            metadata=task.meta,
        )

    def manual_task(self, domain: str, prompt: str, answer: str = "") -> ReasoningTask:
        return ReasoningTask(
            task_id="manual_0001",
            domain=domain,
            prompt=prompt,
            answer=answer,
            goal="Solve the problem",
            meta={"family": domain},
        )

    def build_training_example(self, task: ReasoningTask) -> str:
        state = self.make_state(task)
        return state.serialize() + "\n" + self.build_gold_trace(task)

    def build_gold_trace(self, task: ReasoningTask) -> str:
        tool = task.domain
        thought = {
            "reverse_text": "Reverse the characters in order.",
            "uppercase_text": "Convert every letter to uppercase.",
            "vowel_count": "Count a, e, i, o, and u.",
            "sort_words": "Sort the words lexicographically.",
            "dedupe_words": "Keep the first occurrence of each word.",
        }[task.domain]
        payload = task.prompt.split(":", 1)[-1].strip()
        return (
            f'<action type="THINK">{thought}</action>\n'
            f'<action type="APPLY" tool="{tool}">{payload}</action>\n'
            f'<action type="ANSWER">{task.answer}</action>\n'
            f'<answer>{task.answer}</answer>'
        )

    def build_verifier_examples(self, task: ReasoningTask) -> tuple[str, torch.Tensor, str, torch.Tensor]:
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
        task: ReasoningTask,
        state: ReasoningState,
        local_scores: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        local_scores = local_scores or {}
        has_answer = bool(state.final_answer.strip())
        correct = has_answer and self.evaluate_answer(task, state.final_answer)
        solved = state.status == "solved"

        valid_step = float(local_scores.get("valid_step", 1.0 if solved or has_answer else 0.6))
        structural_progress = min(
            1.0,
            0.14 * len(state.derived_facts)
            + 0.08 * len(state.tool_history)
            + 0.05 * len(state.action_history)
            + 0.04 * len(state.subgoals),
        )
        goal_progress = max(float(local_scores.get("goal_progress", 0.0)), structural_progress)
        if correct:
            goal_progress = max(goal_progress, 0.95)
        elif has_answer:
            goal_progress = min(0.75, max(goal_progress, 0.25))

        proof_completion = 1.0 if correct and solved else (0.2 if has_answer else min(0.2, goal_progress * 0.5))
        risk = float(local_scores.get("risk_score", 0.05 if correct else (0.8 if has_answer else 0.45)))
        branch_priority = max(0.05, min(0.98, 0.58 * goal_progress + 0.22 * valid_step + 0.20 * proof_completion))
        return torch.tensor(
            [
                max(0.02, min(0.99, valid_step)),
                max(0.0, min(0.99, goal_progress)),
                max(0.0, min(0.99, proof_completion)),
                max(0.01, min(0.99, risk)),
                branch_priority,
            ],
            dtype=torch.float32,
        )

    def evaluate_answer(self, task: ReasoningTask, candidate: str) -> bool:
        family = task.meta.get("family", task.domain)
        if family in {"sort_words", "dedupe_words"}:
            return _normalize_whitespace(candidate) == _normalize_whitespace(task.answer)
        return candidate.strip() == task.answer.strip()

    def parse_actions(self, text: str) -> tuple[List[Any], float]:
        return parse_actions(text)

    def fallback_repairs(self, state: ReasoningState) -> List[Action]:
        if state.domain not in GENERATORS:
            return []
        return [Action(type=ActionType.APPLY, tool=state.domain, content=state.problem_text)]

    def render_human_trace(self, state: ReasoningState) -> str:
        return render_human_trace(state)

    def create_executor(self) -> StateExecutor:
        return StateExecutor(StringToolRegistry(), answer_judge=self._answer_judge)

    def _answer_judge(self, state: ReasoningState, candidate: str) -> bool:
        task = ReasoningTask(
            task_id=state.task_id,
            domain=state.domain,
            prompt=state.problem_text,
            answer=state.expected_answer,
            goal=state.goal,
            meta=state.metadata,
        )
        return self.evaluate_answer(task, candidate)

    def maybe_derive_lemma(self, task: ReasoningTask) -> None:
        return None
