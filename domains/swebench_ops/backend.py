from __future__ import annotations

import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from benchmarks.public_catalog import swebench_verified_smoke_suite
from domains.repo_agent_utils import (
    apply_patch_tool as repo_apply_patch_tool,
    create_workspace,
    draft_patch_tool as repo_draft_patch_tool,
    infer_primary_file,
    inspect_tests_tool as repo_inspect_tests_tool,
    list_workspace_files,
    localize_failure_tool as repo_localize_failure_tool,
    read_workspace_file,
    rollback_patch_tool as repo_rollback_patch_tool,
    run_unit_tests_tool as repo_run_unit_tests_tool,
)
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
TMP_ROOT = ROOT / ".tmp-benchmarks" / "swebench"


def _workspace_for(task: ReasoningTask) -> Path:
    return create_workspace(task, TMP_ROOT)


def _list_workspace_files(workspace: Path) -> List[str]:
    return list_workspace_files(workspace)


def _read_workspace_file(workspace: Path, relpath: str) -> str:
    return read_workspace_file(workspace, relpath)


def inspect_workspace(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    files = _list_workspace_files(workspace)
    primary_file = infer_primary_file(files, preferred_file=str(state.metadata.get("primary_file", "")))
    return {
        "ok": True,
        "result": "\n".join(files),
        "goal_progress": 0.15,
        "payload": {
            "files": files,
            "evidence": [f"workspace contains {name}" for name in files[:4]],
            "obligations": ["inspect tests", "run tests", "localize failure"],
            "state_metadata": {
                "workspace_files": files,
                "primary_file": primary_file,
            },
        },
    }


def inspect_tests(arg: str, state: Any = None) -> Dict[str, Any]:
    return repo_inspect_tests_tool(arg, state)


def localize_failure(arg: str, state: Any = None) -> Dict[str, Any]:
    return repo_localize_failure_tool(arg, state)


def read_file(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    relpath = arg.strip() or str(state.metadata.get("primary_file", ""))
    if not relpath:
        files = _list_workspace_files(workspace)
        relpath = infer_primary_file(files)
    if not relpath:
        return {"ok": False, "result": "no file available"}
    text = _read_workspace_file(workspace, relpath)
    resolved = ["inspect source"] if not relpath.startswith("tests/") else ["inspect tests"]
    return {
        "ok": True,
        "result": text,
        "goal_progress": 0.25,
        "payload": {
            "path": relpath,
            "evidence": [f"read {relpath}"],
            "resolved_obligations": resolved,
            "obligations": ["draft patch"] if not relpath.startswith("tests/") else ["inspect source", "draft patch"],
            "state_metadata": {"last_read_file": relpath},
        },
    }


def search_code(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    pattern = arg.strip()
    matches: List[str] = []
    for relpath in _list_workspace_files(workspace):
        text = _read_workspace_file(workspace, relpath)
        for line_no, line in enumerate(text.splitlines(), start=1):
            if pattern and pattern in line:
                matches.append(f"{relpath}:{line_no}:{line.strip()}")
    return {
        "ok": True,
        "result": "\n".join(matches) if matches else "no matches",
        "goal_progress": 0.18,
        "payload": {
            "matches": matches,
            "evidence": matches[:4],
            "resolved_obligations": ["inspect source"] if matches else [],
        },
    }


def run_unit_tests(arg: str, state: Any = None) -> Dict[str, Any]:
    return repo_run_unit_tests_tool(arg, state)


def apply_patch_tool(arg: str, state: Any = None) -> Dict[str, Any]:
    return repo_apply_patch_tool(arg, state)


def draft_patch(arg: str, state: Any = None) -> Dict[str, Any]:
    return repo_draft_patch_tool(arg, state)


def rollback_patch(arg: str, state: Any = None) -> Dict[str, Any]:
    return repo_rollback_patch_tool(arg, state)


class SwebenchToolRegistry:
    def __init__(self) -> None:
        self.tools = {
            "inspect_workspace": inspect_workspace,
            "inspect_tests": inspect_tests,
            "localize_failure": localize_failure,
            "read_file": read_file,
            "search_code": search_code,
            "draft_patch": draft_patch,
            "run_unit_tests": run_unit_tests,
            "apply_patch": apply_patch_tool,
            "rollback_patch": rollback_patch,
        }

    def call(self, name: str, arg: str, state: Any = None) -> Dict[str, Any]:
        fn = self.tools.get(name)
        if fn is None:
            return {"ok": False, "result": f"unknown tool: {name}"}
        try:
            return fn(arg, state)
        except Exception as exc:
            return {"ok": False, "result": f"swebench tool error: {exc}"}


class SwebenchOpsReasoningDomain:
    name = "swebench_ops"
    default_curriculum_config = "config/swebench_ops_curriculum.yaml"

    def __init__(self, runtime_config: Dict[str, Any] | None = None) -> None:
        self._cases = list(swebench_verified_smoke_suite().cases)
        benchmark_cfg = dict((runtime_config or {}).get("benchmark", {}))
        self.assistance_mode = str(benchmark_cfg.get("assistance_mode", "unassisted")).lower()
        self.oracle_hints_enabled = bool(benchmark_cfg.get("oracle_hints_enabled", False))

    def _match_manual_case(self, prompt: str, domain: str) -> Optional[ReasoningTask]:
        text = f"{domain}\n{prompt}".lower()
        score_map: List[tuple[int, ReasoningTask]] = []
        for case in self._cases:
            keywords = set([case.task_id.lower(), case.domain.lower(), str(case.meta.get("family", "")).lower()])
            prompt_bits = case.prompt.lower().replace(".", " ").replace(",", " ").replace(":", " ").split()
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
        return random.choice(self._cases)

    def make_state(self, task: ReasoningTask) -> ReasoningState:
        workspace = _workspace_for(task)
        files = _list_workspace_files(workspace)
        metadata = dict(task.meta)
        metadata["workspace_dir"] = str(workspace)
        metadata["workspace_files"] = files
        metadata["primary_file"] = infer_primary_file(files)
        metadata["benchmark_assistance_mode"] = self.assistance_mode
        if self.assistance_mode == "assisted" and self.oracle_hints_enabled:
            oracle_primary = str(metadata.get("oracle_primary_file", ""))
            if oracle_primary:
                metadata["primary_file"] = oracle_primary
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
            task_id=f"manual_swebench_{uuid.uuid4().hex[:8]}",
            domain=domain,
            prompt=prompt,
            answer=answer,
            goal="Patch the repository so tests pass",
            meta={"family": domain},
        )

    def build_training_example(self, task: ReasoningTask) -> str:
        state = self.make_state(task)
        return state.serialize() + "\n" + self.build_gold_trace(task)

    def build_gold_trace(self, task: ReasoningTask) -> str:
        actions = [
            Action(type=ActionType.THINK, content="inspect workspace and tests, run tests, localize failure, patch the source, and verify"),
            Action(type=ActionType.APPLY, tool="inspect_workspace", content=""),
            Action(type=ActionType.APPLY, tool="inspect_tests", content=""),
            Action(type=ActionType.CHECK, tool="run_unit_tests", content=""),
            Action(type=ActionType.APPLY, tool="localize_failure", content=""),
            Action(type=ActionType.APPLY, tool="read_file", content=str(task.meta.get("oracle_primary_file", ""))),
            Action(type=ActionType.APPLY, tool="draft_patch", content=task.prompt),
            Action(type=ActionType.APPLY, tool="apply_patch", content=""),
            Action(type=ActionType.CHECK, tool="run_unit_tests", content=""),
            Action(type=ActionType.ANSWER, content=task.answer),
        ]
        return render_canonical_actions(actions)

    def build_verifier_examples(self, task: ReasoningTask) -> tuple[str, torch.Tensor, str, torch.Tensor]:
        pos = self.make_state(task)
        pos.final_answer = task.answer
        pos.status = "solved"
        pos.derived_facts.append(task.answer)
        pos.action_history.append({"type": "ANSWER", "content": task.answer})
        pos.tool_history.append({"tool": "run_unit_tests", "result": {"ok": True, "answer": task.answer}})

        neg = self.make_state(task)
        neg.derived_facts.append("tests_not_run")

        pos_t = self.build_verifier_targets(task, pos)
        neg_t = self.build_verifier_targets(task, neg, local_scores={"valid_step": 0.35, "goal_progress": 0.0, "risk_score": 0.85})
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
        valid_step = float(local_scores.get("valid_step", 1.0 if solved or has_answer else 0.5))
        structural_progress = min(
            1.0,
            0.12 * len(state.derived_facts) + 0.10 * len(state.tool_history) + 0.08 * len(state.action_history)
        )
        goal_progress = max(float(local_scores.get("goal_progress", 0.0)), structural_progress)
        if correct:
            goal_progress = max(goal_progress, 0.98)
        proof_completion = 1.0 if correct and solved else (0.25 if has_answer else min(0.2, goal_progress * 0.5))
        risk = float(local_scores.get("risk_score", 0.05 if correct else (0.75 if has_answer else 0.55)))
        branch_priority = max(0.05, min(0.99, 0.58 * goal_progress + 0.22 * valid_step + 0.20 * proof_completion))
        value_estimate = max(0.01, min(0.99, 0.45 * goal_progress + 0.30 * proof_completion + 0.20 * branch_priority + 0.10 * valid_step - 0.15 * risk))
        return torch.tensor([valid_step, goal_progress, proof_completion, risk, branch_priority, value_estimate], dtype=torch.float32)

    def evaluate_answer(self, task: ReasoningTask, candidate: str) -> bool:
        return candidate.strip() == task.answer.strip()

    def parse_actions(self, text: str) -> tuple[List[Any], float]:
        return parse_actions(text)

    def fallback_repairs(self, state: ReasoningState) -> List[Action]:
        tool_names = [record.get("tool", "") for record in state.tool_history if isinstance(record, dict)]
        if "inspect_workspace" not in tool_names:
            return [Action(type=ActionType.APPLY, tool="inspect_workspace", content="")]
        if "inspect_tests" not in tool_names:
            return [Action(type=ActionType.APPLY, tool="inspect_tests", content="")]
        if "run_unit_tests" not in tool_names:
            return [Action(type=ActionType.CHECK, tool="run_unit_tests", content="")]
        if "localize_failure" not in tool_names and state.status != "solved":
            return [Action(type=ActionType.APPLY, tool="localize_failure", content="")]
        if "draft_patch" not in tool_names and state.status != "solved":
            return [Action(type=ActionType.APPLY, tool="draft_patch", content=state.problem_text)]
        if "apply_patch" not in tool_names and state.status != "solved":
            return [Action(type=ActionType.APPLY, tool="apply_patch", content="")]
        if tool_names.count("run_unit_tests") < 2 and state.status != "solved":
            return [Action(type=ActionType.CHECK, tool="run_unit_tests", content="")]
        if state.final_answer.strip():
            return [Action(type=ActionType.ANSWER, content=state.final_answer)]
        return [Action(type=ActionType.BACKTRACK, content="search another repository fix")]

    def allowed_action_types(self, state: ReasoningState) -> List[str]:
        actions = ["THINK", "SUBGOAL", "APPLY", "CHECK"]
        if state.derived_facts or state.tool_history:
            actions.append("ANSWER")
        return actions

    def allowed_tools(self, state: ReasoningState, action_type: str) -> List[str]:
        normalized = action_type.upper()
        if normalized == "CHECK":
            return ["run_unit_tests"]
        if normalized != "APPLY":
            return []
        return ["inspect_workspace", "inspect_tests", "localize_failure", "read_file", "search_code", "draft_patch", "apply_patch", "rollback_patch"]

    def candidate_bindings(self, state: ReasoningState, action_type: str, tool: str = "") -> List[Dict[str, str]]:
        normalized = action_type.upper()
        if normalized == "THINK":
            return [{"content": "inspect tests, run the failure, localize the bug, draft a validated patch, and verify the repository"}]
        if normalized == "SUBGOAL":
            pending = state.obligations[:3] or ["inspect tests", "localize failure", "verify tests"]
            return [{"content": item} for item in pending]
        if normalized == "ANSWER":
            answers = [state.final_answer, str(state.metadata.get("candidate_answer", "")), state.expected_answer]
            return [{"content": item} for item in answers if str(item).strip()]
        if normalized == "CHECK":
            return [{"content": ""}]
        if tool in {"inspect_workspace", "inspect_tests", "localize_failure", "apply_patch", "rollback_patch"}:
            return [{"content": ""}]
        if tool == "read_file":
            paths = [str(state.metadata.get("primary_file", ""))]
            paths.extend([name for name in state.metadata.get("workspace_files", []) if name.startswith("tests/")])
            return [{"content": path} for path in paths if path][:3]
        if tool == "search_code":
            symbols = list(state.metadata.get("test_symbols", []))
            return [{"content": symbol} for symbol in symbols[:3]] or [{"content": "return"}]
        if tool == "draft_patch":
            context = str(state.metadata.get("primary_file", "")) or state.problem_text
            return [{"content": context}]
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
            "Emit canonical JSON actions. Solve the repository task without oracle patch hints.\n"
            'ACTION {"type":"APPLY","tool":"inspect_tests","content":""}\n'
            'ACTION {"type":"CHECK","tool":"run_unit_tests","content":""}\n'
            'ACTION {"type":"APPLY","tool":"localize_failure","content":""}\n'
            'ACTION {"type":"APPLY","tool":"draft_patch","content":"bug context"}\n'
            'ACTION {"type":"ANSWER","content":"patched_and_verified"}'
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
                tool_names=self.allowed_tools(state, "APPLY") + self.allowed_tools(state, "CHECK"),
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
        return " || ".join(
            [
                state.domain,
                str(state.metadata.get("primary_file", "")),
                " | ".join(state.derived_facts[-3:]),
                " | ".join(state.obligations[-3:]),
                state.final_answer.strip(),
            ]
        )

    def render_human_trace(self, state: ReasoningState) -> str:
        return render_human_trace(state)

    def create_executor(self) -> StateExecutor:
        return StateExecutor(SwebenchToolRegistry(), answer_judge=self._answer_judge)

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
