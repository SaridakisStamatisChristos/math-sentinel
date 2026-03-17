from __future__ import annotations

import itertools
import random
import re
from typing import Any, Dict, List, Optional, Sequence

import torch

from engine.action_format import render_canonical_actions
from engine.actions import Action, ActionType
from engine.executor import StateExecutor
from engine.prompting import build_search_prompt
from engine.state import ReasoningState
from engine.task import ReasoningTask
from engine.traces import render_human_trace
from memory.retrieval import retrieve_context
from proof.parser import parse_actions


TASK_NAMES = ["design", "build", "test", "review", "deploy", "pack", "shop", "cook"]
ITEM_NAMES = ["bread", "milk", "fruit", "notebook", "cable", "tea"]


def _rid(prefix: str) -> str:
    return f"{prefix}_{random.randint(10**7, 10**8 - 1)}"


def _format_plan(steps: Sequence[str]) -> str:
    return " -> ".join(steps) if steps else "none"


def _normalize_plan(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    cleaned = cleaned.replace(" ,", ",").replace(", ", ",")
    cleaned = cleaned.replace(" -> ", "->").replace(" ->", "->").replace("-> ", "->")
    return cleaned


def _parse_task_lines(text: str) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"- (?P<name>[a-z_]+) \(duration=(?P<duration>\d+), priority=(?P<priority>\d+), deps=(?P<deps>[^)]+)\)"
    )
    for match in pattern.finditer(text):
        deps_raw = match.group("deps").strip()
        deps = [] if deps_raw == "none" else [dep.strip() for dep in deps_raw.split(",") if dep.strip()]
        tasks.append(
            {
                "name": match.group("name"),
                "duration": int(match.group("duration")),
                "priority": int(match.group("priority")),
                "deps": deps,
            }
        )
    return tasks


def _parse_item_lines(text: str) -> tuple[int, List[Dict[str, Any]]]:
    budget_match = re.search(r"budget=(?P<budget>\d+)", text)
    budget = int(budget_match.group("budget")) if budget_match else 0
    items: List[Dict[str, Any]] = []
    pattern = re.compile(r"- (?P<name>[a-z_]+) \(cost=(?P<cost>\d+), priority=(?P<priority>\d+)\)")
    for match in pattern.finditer(text):
        items.append(
            {
                "name": match.group("name"),
                "cost": int(match.group("cost")),
                "priority": int(match.group("priority")),
            }
        )
    return budget, items


def _topological_plan(tasks: List[Dict[str, Any]]) -> List[str]:
    remaining = {task["name"]: list(task["deps"]) for task in tasks}
    order: List[str] = []
    while remaining:
        available = [name for name, deps in remaining.items() if all(dep in order for dep in deps)]
        if not available:
            break
        next_task = sorted(available)[0]
        order.append(next_task)
        remaining.pop(next_task)
    return order


def _best_shopping_plan(budget: int, items: List[Dict[str, Any]]) -> List[str]:
    best: tuple[int, int, int, List[str]] | None = None
    for mask in range(1 << len(items)):
        picked = [items[idx] for idx in range(len(items)) if mask & (1 << idx)]
        total_cost = sum(item["cost"] for item in picked)
        if total_cost > budget:
            continue
        total_priority = sum(item["priority"] for item in picked)
        candidate_names = [item["name"] for item in picked]
        score = (total_priority, len(candidate_names), -total_cost, candidate_names)
        if best is None or score > best:
            best = score
    return best[3] if best is not None else []


def _is_valid_order(order: Sequence[Dict[str, Any]], limit: int) -> bool:
    seen: set[str] = set()
    duration = 0
    for task in order:
        if any(dep not in seen for dep in task["deps"]):
            return False
        duration += task["duration"]
        if duration > limit:
            return False
        seen.add(task["name"])
    return True


def _best_day_plan(limit: int, tasks: List[Dict[str, Any]]) -> List[str]:
    best: tuple[int, int, int, List[str]] | None = None
    for size in range(1, len(tasks) + 1):
        for subset in itertools.permutations(tasks, size):
            if not _is_valid_order(subset, limit):
                continue
            total_priority = sum(task["priority"] for task in subset)
            total_duration = sum(task["duration"] for task in subset)
            names = [task["name"] for task in subset]
            score = (total_priority, len(names), -total_duration, names)
            if best is None or score > best:
                best = score
    return best[3] if best is not None else []


def gen_project_plan() -> ReasoningTask:
    chosen = random.sample(TASK_NAMES, 3)
    tasks = [
        {"name": chosen[0], "duration": 1, "priority": 3, "deps": []},
        {"name": chosen[1], "duration": 2, "priority": 4, "deps": [chosen[0]]},
        {"name": chosen[2], "duration": 1, "priority": 2, "deps": [chosen[1]]},
    ]
    prompt = (
        "Create a valid project plan.\nTasks:\n"
        + "\n".join(
            f"- {task['name']} (duration={task['duration']}, priority={task['priority']}, deps={','.join(task['deps']) or 'none'})"
            for task in tasks
        )
        + "\nReturn the ordered task plan."
    )
    return ReasoningTask(
        task_id=_rid("plan_project"),
        domain="project_plan",
        prompt=prompt,
        answer=_format_plan(_topological_plan(tasks)),
        goal="Return a valid dependency-respecting order",
        meta={"family": "project_plan", "tasks": tasks},
    )


def gen_shopping_plan() -> ReasoningTask:
    budget = random.randint(5, 8)
    chosen = random.sample(ITEM_NAMES, 4)
    items = [
        {"name": chosen[0], "cost": 2, "priority": 5},
        {"name": chosen[1], "cost": 3, "priority": 4},
        {"name": chosen[2], "cost": 4, "priority": 6},
        {"name": chosen[3], "cost": 1, "priority": 2},
    ]
    prompt = (
        f"Choose the best shopping plan under budget={budget}.\nItems:\n"
        + "\n".join(f"- {item['name']} (cost={item['cost']}, priority={item['priority']})" for item in items)
        + "\nReturn the chosen items in input order separated by commas."
    )
    answer = ", ".join(_best_shopping_plan(budget, items))
    return ReasoningTask(
        task_id=_rid("plan_shop"),
        domain="shopping_plan",
        prompt=prompt,
        answer=answer,
        goal="Maximize total priority without exceeding the budget",
        meta={"family": "shopping_plan", "items": items, "budget": budget},
    )


def gen_day_plan() -> ReasoningTask:
    chosen = random.sample(TASK_NAMES, 3)
    limit = random.randint(3, 5)
    tasks = [
        {"name": chosen[0], "duration": 1, "priority": 2, "deps": []},
        {"name": chosen[1], "duration": 2, "priority": 4, "deps": [chosen[0]]},
        {"name": chosen[2], "duration": 2, "priority": 3, "deps": []},
    ]
    prompt = (
        f"Create the best day plan under time_limit={limit}.\nTasks:\n"
        + "\n".join(
            f"- {task['name']} (duration={task['duration']}, priority={task['priority']}, deps={','.join(task['deps']) or 'none'})"
            for task in tasks
        )
        + "\nReturn the ordered task plan."
    )
    return ReasoningTask(
        task_id=_rid("plan_day"),
        domain="day_plan",
        prompt=prompt,
        answer=_format_plan(_best_day_plan(limit, tasks)),
        goal="Maximize priority while respecting dependencies and time",
        meta={"family": "day_plan", "tasks": tasks, "time_limit": limit},
    )


GENERATORS = {
    "project_plan": gen_project_plan,
    "shopping_plan": gen_shopping_plan,
    "day_plan": gen_day_plan,
}


def sample_task(domains: List[str]) -> ReasoningTask:
    domain = random.choice(domains)
    return GENERATORS[domain]()


def project_plan(arg: str, state: Any = None) -> Dict[str, Any]:
    tasks = _parse_task_lines(arg)
    rendered = _format_plan(_topological_plan(tasks))
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}


def shopping_plan(arg: str, state: Any = None) -> Dict[str, Any]:
    budget, items = _parse_item_lines(arg)
    rendered = ", ".join(_best_shopping_plan(budget, items))
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}


def day_plan(arg: str, state: Any = None) -> Dict[str, Any]:
    limit_match = re.search(r"time_limit=(?P<limit>\d+)", arg)
    limit = int(limit_match.group("limit")) if limit_match else 0
    tasks = _parse_task_lines(arg)
    rendered = _format_plan(_best_day_plan(limit, tasks))
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}


class PlanningToolRegistry:
    def __init__(self) -> None:
        self.tools = {
            "project_plan": project_plan,
            "shopping_plan": shopping_plan,
            "day_plan": day_plan,
        }

    def call(self, name: str, arg: str, state: Any = None) -> Dict[str, Any]:
        fn = self.tools.get(name)
        if fn is None:
            return {"ok": False, "result": f"unknown tool: {name}"}
        return fn(arg, state)


class PlanningOpsReasoningDomain:
    name = "planning_ops"
    default_curriculum_config = "config/planning_ops_curriculum.yaml"

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
        actions = [
            Action(type=ActionType.THINK, content="Plan around dependencies, priorities, and constraints."),
            Action(type=ActionType.SUBGOAL, content="derive feasible plan"),
            Action(type=ActionType.APPLY, tool=task.domain, content=task.prompt),
            Action(type=ActionType.RESOLVE_SUBGOAL, content="derive feasible plan"),
            Action(type=ActionType.ANSWER, content=task.answer),
        ]
        return render_canonical_actions(actions)

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

        valid_step = float(local_scores.get("valid_step", 1.0 if solved or has_answer else 0.55))
        structural_progress = min(
            1.0,
            0.12 * len(state.derived_facts)
            + 0.10 * len(state.tool_history)
            + 0.08 * len(state.action_history)
            + 0.08 * len(state.subgoals),
        )
        goal_progress = max(float(local_scores.get("goal_progress", 0.0)), structural_progress)
        if correct:
            goal_progress = max(goal_progress, 0.95)
        elif has_answer:
            goal_progress = min(0.75, max(goal_progress, 0.3))

        proof_completion = 1.0 if correct and solved else (0.22 if has_answer else min(0.2, goal_progress * 0.5))
        risk = float(local_scores.get("risk_score", 0.05 if correct else (0.82 if has_answer else 0.45)))
        branch_priority = max(0.05, min(0.98, 0.56 * goal_progress + 0.22 * valid_step + 0.22 * proof_completion))
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
        return _normalize_plan(candidate) == _normalize_plan(task.answer)

    def parse_actions(self, text: str) -> tuple[List[Any], float]:
        return parse_actions(text)

    def fallback_repairs(self, state: ReasoningState) -> List[Action]:
        if state.domain not in GENERATORS:
            return []
        return [Action(type=ActionType.APPLY, tool=state.domain, content=state.problem_text)]

    def action_format_instructions(self) -> str:
        return (
            "Emit one JSON action per line in canonical form.\n"
            "Use SUBGOAL and RESOLVE_SUBGOAL when planning intermediate steps.\n"
            'ACTION {"type":"THINK","content":"plan"}\n'
            'ACTION {"type":"SUBGOAL","content":"..."}\n'
            'ACTION {"type":"APPLY","tool":"tool_name","content":"arguments"}\n'
            'ACTION {"type":"ANSWER","content":"final answer"}'
        )

    def build_search_prompt(
        self,
        state: ReasoningState,
        *,
        lemma_store: Any | None = None,
        hard_case_store: Any | None = None,
        tactic_stats: Any | None = None,
    ) -> str:
        retrieval_context = None
        if lemma_store is not None and hard_case_store is not None:
            retrieval_context = retrieve_context(lemma_store, hard_case_store, state.domain, state.problem_text)
        tactic_hints = None
        if tactic_stats is not None:
            ranked = tactic_stats.top_tactics(state.domain, limit=3)
            tactic_hints = [f"{name} bias={bias:.2f}" for name, bias in ranked if bias != 0.5]
        return build_search_prompt(
            state,
            self.action_format_instructions(),
            retrieval_context=retrieval_context,
            tactic_hints=tactic_hints,
        )

    def state_signature(self, state: ReasoningState) -> str:
        derived = " | ".join(state.derived_facts[-3:])
        subgoals = " | ".join(state.subgoals)
        return " || ".join([state.domain, state.problem_text, derived, subgoals, state.final_answer.strip()])

    def render_human_trace(self, state: ReasoningState) -> str:
        return render_human_trace(state)

    def create_executor(self) -> StateExecutor:
        return StateExecutor(PlanningToolRegistry(), answer_judge=self._answer_judge)

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

    def benchmark_tasks(self) -> List[ReasoningTask]:
        return [
            self.manual_task(
                "project_plan",
                "Create a valid project plan.\nTasks:\n- design (duration=1, priority=3, deps=none)\n- build (duration=2, priority=4, deps=design)\n- test (duration=1, priority=2, deps=build)\nReturn the ordered task plan.",
                "design -> build -> test",
            ),
            self.manual_task(
                "shopping_plan",
                "Choose the best shopping plan under budget=5.\nItems:\n- bread (cost=2, priority=5)\n- milk (cost=3, priority=4)\n- fruit (cost=4, priority=6)\n- tea (cost=1, priority=2)\nReturn the chosen items in input order separated by commas.",
                "bread, milk",
            ),
            self.manual_task(
                "day_plan",
                "Create the best day plan under time_limit=3.\nTasks:\n- design (duration=1, priority=2, deps=none)\n- build (duration=2, priority=4, deps=design)\n- review (duration=2, priority=3, deps=none)\nReturn the ordered task plan.",
                "design -> build",
            ),
        ]
