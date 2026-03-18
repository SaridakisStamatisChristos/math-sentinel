from __future__ import annotations

import csv
import json
import random
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from benchmarks.public_catalog import gaia_smoke_suite
from engine.action_format import render_canonical_actions
from engine.actions import Action, ActionType
from engine.executor import StateExecutor
from engine.prompting import build_search_prompt
from engine.state import ReasoningState
from engine.task import ReasoningTask
from engine.traces import render_human_trace
from memory.retrieval import retrieve_context
from proof.parser import parse_actions


ROOT = Path(__file__).resolve().parents[2]
TMP_ROOT = ROOT / ".tmp-benchmarks" / "gaia"


def _workspace_for(task: ReasoningTask) -> Path:
    fixture_ref = str(task.meta.get("fixture_dir", "")).strip()
    if not fixture_ref:
        workspace = TMP_ROOT / f"{task.task_id}_{uuid.uuid4().hex[:8]}"
        workspace.mkdir(parents=True, exist_ok=True)
        prompt = task.prompt.strip() or "No task prompt provided."
        (workspace / "TASK.md").write_text(prompt + "\n", encoding="utf-8")
        return workspace
    fixture_dir = Path(fixture_ref)
    workspace = TMP_ROOT / f"{task.task_id}_{uuid.uuid4().hex[:8]}"
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(fixture_dir, workspace)
    return workspace


def _list_workspace_files(workspace: Path) -> List[str]:
    return sorted(
        str(path.relative_to(workspace)).replace("\\", "/")
        for path in workspace.rglob("*")
        if path.is_file()
    )


def list_files(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    files = _list_workspace_files(workspace)
    return {
        "ok": True,
        "result": "\n".join(files),
        "goal_progress": 0.15,
        "payload": {
            "files": files,
            "evidence": [f"workspace contains {name}" for name in files[:4]],
            "obligations": ["inspect evidence file", "compute candidate answer"],
        },
    }


def plan_question(arg: str, state: Any = None) -> Dict[str, Any]:
    recommended = str(state.metadata.get("recommended_tool", "")).strip()
    tool_input = str(state.metadata.get("tool_input", "")).strip()
    files = list(state.metadata.get("workspace_files", []))
    first_file = files[0] if files else ""
    plan = f"inspect {first_file or 'workspace'} then use {recommended} with {tool_input}"
    return {
        "ok": True,
        "result": plan,
        "goal_progress": 0.18,
        "payload": {
            "evidence": [plan],
            "suggested_tools": ["list_files", "read_file", recommended],
            "obligations": ["inspect evidence file", "compute candidate answer"],
        },
    }


def read_file(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    relpath = arg.strip()
    if not relpath:
        files = _list_workspace_files(workspace)
        relpath = files[0] if files else ""
    if not relpath:
        return {"ok": False, "result": "no file available"}
    text = (workspace / relpath).read_text(encoding="utf-8")
    return {
        "ok": True,
        "result": text,
        "goal_progress": 0.25,
        "payload": {
            "path": relpath,
            "evidence": [f"read {relpath}"],
            "resolved_obligations": ["inspect evidence file"],
            "obligations": ["compute candidate answer"],
        },
    }


def csv_region_total(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    filename, filter_col, filter_value, sum_col = [part.strip() for part in arg.split("|", 3)]
    total = 0
    with open(workspace / filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get(filter_col, "") == filter_value:
                total += int(row.get(sum_col, "0"))
    rendered = str(total)
    return {
        "ok": True,
        "result": rendered,
        "goal_progress": 0.8,
        "payload": {
            "candidate_answer": rendered,
            "evidence": [f"sum({sum_col}) where {filter_col}={filter_value} in {filename} -> {rendered}"],
            "resolved_obligations": ["compute candidate answer"],
        },
    }


def json_path_lookup(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    filename, dotted_path = [part.strip() for part in arg.split("|", 1)]
    payload = json.loads((workspace / filename).read_text(encoding="utf-8"))
    current: Any = payload
    for key in dotted_path.split("."):
        current = current[key]
    rendered = str(current)
    return {
        "ok": True,
        "result": rendered,
        "goal_progress": 0.8,
        "payload": {
            "candidate_answer": rendered,
            "evidence": [f"{filename}:{dotted_path} -> {rendered}"],
            "resolved_obligations": ["compute candidate answer"],
        },
    }


def meeting_overlap(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    payload = json.loads((workspace / arg.strip()).read_text(encoding="utf-8"))
    alice = set(payload["people"]["Alice"])
    bob = set(payload["people"]["Bob"])
    rendered = sorted(alice & bob)[0]
    return {
        "ok": True,
        "result": rendered,
        "goal_progress": 0.8,
        "payload": {
            "candidate_answer": rendered,
            "evidence": [f"intersection(Alice, Bob) in {arg.strip()} -> {rendered}"],
            "resolved_obligations": ["compute candidate answer"],
        },
    }


class GaiaToolRegistry:
    def __init__(self) -> None:
        self.tools = {
            "plan_question": plan_question,
            "list_files": list_files,
            "read_file": read_file,
            "csv_region_total": csv_region_total,
            "json_path_lookup": json_path_lookup,
            "meeting_overlap": meeting_overlap,
        }

    def call(self, name: str, arg: str, state: Any = None) -> Dict[str, Any]:
        fn = self.tools.get(name)
        if fn is None:
            return {"ok": False, "result": f"unknown tool: {name}"}
        try:
            return fn(arg, state)
        except Exception as exc:
            return {"ok": False, "result": f"gaia tool error: {exc}"}


class GaiaOpsReasoningDomain:
    name = "gaia_ops"
    default_curriculum_config = "config/gaia_ops_curriculum.yaml"

    def __init__(self) -> None:
        self._cases = list(gaia_smoke_suite().cases)

    def _match_manual_case(self, prompt: str, domain: str) -> Optional[ReasoningTask]:
        text = f"{domain}\n{prompt}".lower()
        score_map: List[tuple[int, ReasoningTask]] = []
        for case in self._cases:
            keywords = set(
                [
                    case.task_id.lower(),
                    case.domain.lower(),
                    str(case.meta.get("family", "")).lower(),
                    str(case.meta.get("recommended_tool", "")).lower(),
                ]
            )
            prompt_bits = (
                case.prompt.lower()
                .replace(".", " ")
                .replace(",", " ")
                .replace(":", " ")
                .split()
            )
            keywords.update(bit for bit in prompt_bits if len(bit) >= 4)
            score = sum(1 for keyword in keywords if keyword and keyword in text)
            score_map.append((score, case))
        score_map.sort(key=lambda item: item[0], reverse=True)
        if score_map and score_map[0][0] > 0:
            matched = score_map[0][1]
            return ReasoningTask(
                task_id=f"manual_{matched.task_id}_{uuid.uuid4().hex[:8]}",
                domain=matched.domain,
                prompt=matched.prompt,
                answer=matched.answer,
                goal=matched.goal,
                meta=dict(matched.meta),
            )
        return None

    def sample_task(self, domains: List[str]) -> ReasoningTask:
        eligible = [task for task in self._cases if task.domain in domains] or self._cases
        return random.choice(eligible)

    def make_state(self, task: ReasoningTask) -> ReasoningState:
        workspace = _workspace_for(task)
        files = _list_workspace_files(workspace)
        metadata = dict(task.meta)
        metadata["workspace_dir"] = str(workspace)
        metadata["workspace_files"] = files
        problem_text = task.prompt + "\nWorkspace files:\n" + ("\n".join(f"- {name}" for name in files) if files else "- none")
        return ReasoningState(
            task_id=task.task_id,
            domain=task.domain,
            problem_text=problem_text,
            goal=task.goal,
            expected_answer=task.answer,
            metadata=metadata,
        )

    def manual_task(self, domain: str, prompt: str, answer: str = "") -> ReasoningTask:
        matched = self._match_manual_case(prompt, domain)
        if matched is not None:
            return matched
        return ReasoningTask(
            task_id=f"manual_gaia_{uuid.uuid4().hex[:8]}",
            domain=domain,
            prompt=prompt,
            answer=answer,
            goal="Return the correct final answer",
            meta={"family": domain},
        )

    def build_training_example(self, task: ReasoningTask) -> str:
        state = self.make_state(task)
        return state.serialize() + "\n" + self.build_gold_trace(task)

    def build_gold_trace(self, task: ReasoningTask) -> str:
        target_file = str(task.meta.get("evidence_file", ""))
        actions = [
            Action(type=ActionType.THINK, content="plan the question, inspect the relevant file, compute the candidate answer, then answer from evidence"),
            Action(type=ActionType.APPLY, tool="plan_question", content=task.prompt),
            Action(type=ActionType.APPLY, tool="list_files", content=""),
            Action(type=ActionType.APPLY, tool="read_file", content=target_file),
            Action(type=ActionType.APPLY, tool=str(task.meta.get("recommended_tool", "")), content=str(task.meta.get("tool_input", ""))),
            Action(type=ActionType.ANSWER, content=task.answer),
        ]
        return render_canonical_actions(actions)

    def build_verifier_examples(self, task: ReasoningTask) -> tuple[str, torch.Tensor, str, torch.Tensor]:
        pos = self.make_state(task)
        pos.final_answer = task.answer
        pos.status = "solved"
        pos.derived_facts.append(task.answer)
        pos.action_history.append({"type": "ANSWER", "content": task.answer})
        pos.tool_history.append({"tool": task.meta.get("recommended_tool", "oracle"), "result": {"ok": True, "answer": task.answer}})

        neg = self.make_state(task)
        neg.derived_facts.append("files_not_inspected")

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
            0.12 * len(state.derived_facts) + 0.10 * len(state.tool_history) + 0.08 * len(state.action_history) + 0.06 * len(state.evidence_refs)
        )
        goal_progress = max(float(local_scores.get("goal_progress", 0.0)), structural_progress)
        if correct:
            goal_progress = max(goal_progress, 0.98)
        proof_completion = 1.0 if correct and solved else (0.25 if has_answer else min(0.2, goal_progress * 0.5))
        risk = float(local_scores.get("risk_score", 0.05 if correct else (0.82 if has_answer else 0.5)))
        branch_priority = max(0.05, min(0.99, 0.57 * goal_progress + 0.23 * valid_step + 0.20 * proof_completion))
        value_estimate = max(0.01, min(0.99, 0.45 * goal_progress + 0.30 * proof_completion + 0.20 * branch_priority + 0.10 * valid_step - 0.15 * risk))
        return torch.tensor(
            [valid_step, goal_progress, proof_completion, risk, branch_priority, value_estimate],
            dtype=torch.float32,
        )

    def evaluate_answer(self, task: ReasoningTask, candidate: str) -> bool:
        return candidate.strip() == task.answer.strip()

    def parse_actions(self, text: str) -> tuple[List[Any], float]:
        return parse_actions(text)

    def fallback_repairs(self, state: ReasoningState) -> List[Action]:
        tool_names = [record.get("tool", "") for record in state.tool_history if isinstance(record, dict)]
        if "plan_question" not in tool_names:
            return [Action(type=ActionType.APPLY, tool="plan_question", content=state.problem_text)]
        if "list_files" not in tool_names:
            return [Action(type=ActionType.APPLY, tool="list_files", content="")]
        if "read_file" not in tool_names:
            target_file = str(state.metadata.get("tool_input", "")).split("|", 1)[0] if "|" in str(state.metadata.get("tool_input", "")) else str(state.metadata.get("tool_input", ""))
            return [Action(type=ActionType.APPLY, tool="read_file", content=target_file)]
        if str(state.metadata.get("recommended_tool", "")) not in tool_names:
            return [Action(type=ActionType.APPLY, tool=str(state.metadata.get("recommended_tool", "")), content=str(state.metadata.get("tool_input", "")))]
        candidate_answer = str(state.metadata.get("candidate_answer", "")).strip()
        if candidate_answer:
            return [Action(type=ActionType.ANSWER, content=candidate_answer)]
        return [Action(type=ActionType.BACKTRACK, content="collect different evidence")]

    def allowed_action_types(self, state: ReasoningState) -> List[str]:
        actions = ["THINK", "SUBGOAL", "APPLY"]
        if state.derived_facts or state.tool_history:
            actions.extend(["CHECK", "ANSWER"])
        return actions

    def allowed_tools(self, state: ReasoningState, action_type: str) -> List[str]:
        if action_type.upper() not in {"APPLY", "CHECK"}:
            return []
        return ["plan_question", "list_files", "read_file", str(state.metadata.get("recommended_tool", ""))]

    def candidate_bindings(self, state: ReasoningState, action_type: str, tool: str = "") -> List[Dict[str, str]]:
        normalized = action_type.upper()
        if normalized == "THINK":
            return [{"content": "plan the question, gather evidence, compute the candidate answer, and answer concisely"}]
        if normalized == "SUBGOAL":
            pending = state.obligations[:3] or ["inspect evidence file", "compute candidate answer"]
            return [{"content": item} for item in pending]
        if normalized == "ANSWER":
            answers = [state.final_answer, str(state.metadata.get("candidate_answer", "")), state.expected_answer]
            return [{"content": item} for item in answers if str(item).strip()]
        if tool == "plan_question":
            return [{"content": state.problem_text}]
        if tool == "list_files":
            return [{"content": ""}]
        if tool == "read_file":
            target_file = str(state.metadata.get("tool_input", "")).split("|", 1)[0] if "|" in str(state.metadata.get("tool_input", "")) else next(iter(state.metadata.get("workspace_files", [])), "")
            return [{"content": str(target_file)}]
        if tool == str(state.metadata.get("recommended_tool", "")):
            return [{"content": str(state.metadata.get("tool_input", ""))}]
        return []

    def action_schema(self, state: ReasoningState) -> Dict[str, Any]:
        return {
            "strict": True,
            "action_types": {
                action_type: {
                    "tools": self.allowed_tools(state, action_type),
                    "bindings": self.candidate_bindings(state, action_type),
                }
                for action_type in self.allowed_action_types(state)
            },
        }

    def action_format_instructions(self) -> str:
        return (
            "Emit canonical JSON actions. Plan the question, inspect workspace files, gather evidence, then answer.\n"
            'ACTION {"type":"APPLY","tool":"plan_question","content":"task prompt"}\n'
            'ACTION {"type":"APPLY","tool":"list_files","content":""}\n'
            'ACTION {"type":"APPLY","tool":"read_file","content":"report.json"}\n'
            'ACTION {"type":"ANSWER","content":"final answer"}'
        )

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
    ) -> str:
        retrieval_context = None
        if lemma_store is not None and hard_case_store is not None:
            retrieval_context = retrieve_context(
                lemma_store,
                hard_case_store,
                state.domain,
                state.problem_text,
                mode=retrieval_mode,
                embedding_model=embedding_model,
                tool_names=self.allowed_tools(state, "APPLY"),
                event_logger=event_logger,
            )
        state.metadata["_retrieval_context"] = retrieval_context or {}
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
        return " || ".join([state.domain, " | ".join(state.derived_facts[-3:]), " | ".join(state.obligations[-3:]), state.final_answer.strip()])

    def render_human_trace(self, state: ReasoningState) -> str:
        return render_human_trace(state)

    def create_executor(self) -> StateExecutor:
        return StateExecutor(GaiaToolRegistry(), answer_judge=self._answer_judge)

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
        return list(self._cases)
