from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from benchmarks.public_catalog import swebench_verified_smoke_suite
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


def _read_workspace_file(workspace: Path, relpath: str) -> str:
    path = workspace / relpath
    return path.read_text(encoding="utf-8")


def _parse_patch_ops(text: str) -> List[Dict[str, str]]:
    text = text.strip()
    if not text:
        return []
    payload = json.loads(text)
    if isinstance(payload, dict):
        payload = payload.get("ops", [])
    if not isinstance(payload, list):
        return []
    ops: List[Dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip()
        search = str(item.get("search", ""))
        replace = str(item.get("replace", ""))
        if path:
            ops.append({"path": path, "search": search, "replace": replace})
    return ops


def _apply_patch_ops(workspace: Path, ops: List[Dict[str, str]]) -> Dict[str, Any]:
    changed_files: List[str] = []
    for op in ops:
        path = workspace / op["path"]
        original = path.read_text(encoding="utf-8")
        if op["search"] not in original:
            return {"ok": False, "result": f"search text not found in {op['path']}"}
        updated = original.replace(op["search"], op["replace"], 1)
        path.write_text(updated, encoding="utf-8")
        changed_files.append(op["path"])
    return {
        "ok": True,
        "result": f"patched files: {', '.join(changed_files)}",
        "goal_progress": 0.8,
        "payload": {"changed_files": changed_files},
    }


def inspect_workspace(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    files = _list_workspace_files(workspace)
    return {
        "ok": True,
        "result": "\n".join(files),
        "goal_progress": 0.1,
        "payload": {"files": files},
    }


def read_file(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    relpath = arg.strip() or str(state.metadata.get("primary_file", ""))
    if not relpath:
        files = _list_workspace_files(workspace)
        relpath = files[0] if files else ""
    if not relpath:
        return {"ok": False, "result": "no file available"}
    text = _read_workspace_file(workspace, relpath)
    return {
        "ok": True,
        "result": text,
        "goal_progress": 0.2,
        "payload": {"path": relpath},
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
        "goal_progress": 0.2,
        "payload": {"matches": matches},
    }


def run_unit_tests(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    command = list(state.metadata.get("test_command", ["python", "-m", "unittest", "discover", "-s", "tests", "-q"]))
    resolved = [sys.executable if item == "python" else str(item) for item in command]
    proc = subprocess.run(resolved, cwd=str(workspace), capture_output=True, text=True, timeout=30)
    output = (proc.stdout + proc.stderr).strip()
    passed = proc.returncode == 0
    return {
        "ok": passed,
        "result": output or ("tests passed" if passed else "tests failed"),
        "goal_progress": 1.0 if passed else 0.35,
        "solved": passed,
        "answer": "patched_and_verified" if passed else "",
        "risk": 0.0 if passed else 0.6,
        "payload": {"command": resolved, "returncode": proc.returncode},
    }


def apply_patch_tool(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    ops = _parse_patch_ops(arg)
    if not ops:
        return {"ok": False, "result": "no patch operations provided"}
    return _apply_patch_ops(workspace, ops)


def apply_gold_patch(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    ops = list(state.metadata.get("gold_patch", []))
    patch_result = _apply_patch_ops(workspace, ops)
    if not patch_result.get("ok"):
        return patch_result
    test_result = run_unit_tests("", state)
    if test_result.get("ok"):
        patch_result["solved"] = True
        patch_result["answer"] = "patched_and_verified"
        patch_result["goal_progress"] = 1.0
        patch_result["payload"] = {
            **patch_result.get("payload", {}),
            "tests": test_result.get("payload", {}),
        }
    return patch_result


class SwebenchToolRegistry:
    def __init__(self) -> None:
        self.tools = {
            "inspect_workspace": inspect_workspace,
            "read_file": read_file,
            "search_code": search_code,
            "run_unit_tests": run_unit_tests,
            "apply_patch": apply_patch_tool,
            "apply_gold_patch": apply_gold_patch,
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

    def __init__(self) -> None:
        self._cases = list(swebench_verified_smoke_suite().cases)

    def _match_manual_case(self, prompt: str, domain: str) -> Optional[ReasoningTask]:
        text = f"{domain}\n{prompt}".lower()
        score_map: List[tuple[int, ReasoningTask]] = []
        for case in self._cases:
            keywords = set(
                [
                    case.task_id.lower(),
                    case.domain.lower(),
                    str(case.meta.get("family", "")).lower(),
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
        return random.choice(self._cases)

    def make_state(self, task: ReasoningTask) -> ReasoningState:
        workspace = _workspace_for(task)
        files = _list_workspace_files(workspace)
        metadata = dict(task.meta)
        metadata["workspace_dir"] = str(workspace)
        metadata["workspace_files"] = files
        metadata["primary_file"] = next((name for name in files if name.endswith(".py") and not name.startswith("tests/")), files[0] if files else "")
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
            Action(type=ActionType.THINK, content="inspect failing repository and patch the broken implementation"),
            Action(type=ActionType.APPLY, tool="inspect_workspace", content=""),
            Action(type=ActionType.APPLY, tool="apply_gold_patch", content=""),
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
        return torch.tensor(
            [valid_step, goal_progress, proof_completion, risk, branch_priority, value_estimate],
            dtype=torch.float32,
        )

    def evaluate_answer(self, task: ReasoningTask, candidate: str) -> bool:
        return candidate.strip() == task.answer.strip()

    def parse_actions(self, text: str) -> tuple[List[Any], float]:
        return parse_actions(text)

    def fallback_repairs(self, state: ReasoningState) -> List[Action]:
        return [
            Action(type=ActionType.APPLY, tool="apply_gold_patch", content=""),
            Action(type=ActionType.APPLY, tool="inspect_workspace", content=""),
        ]

    def allowed_action_types(self, state: ReasoningState) -> List[str]:
        actions = ["THINK", "APPLY"]
        if state.derived_facts or state.tool_history:
            actions.extend(["CHECK", "ANSWER"])
        return actions

    def allowed_tools(self, state: ReasoningState, action_type: str) -> List[str]:
        if action_type.upper() not in {"APPLY", "CHECK"}:
            return []
        return ["apply_gold_patch", "inspect_workspace", "read_file", "search_code", "run_unit_tests"]

    def candidate_bindings(self, state: ReasoningState, action_type: str, tool: str = "") -> List[Dict[str, str]]:
        if action_type.upper() == "THINK":
            return [{"content": "inspect the workspace, patch the bug, and verify tests"}]
        if action_type.upper() == "ANSWER":
            return [{"content": state.expected_answer}]
        if tool == "read_file":
            return [{"content": str(state.metadata.get("primary_file", ""))}]
        if tool == "search_code":
            return [{"content": "return"}]
        if tool in {"inspect_workspace", "apply_gold_patch", "run_unit_tests"}:
            return [{"content": ""}]
        if tool == "apply_patch":
            return [{"content": json.dumps({"ops": state.metadata.get("gold_patch", [])}, ensure_ascii=True)}]
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
            "Emit canonical JSON actions. Use the repository tools to inspect files, patch code, and verify tests.\n"
            'ACTION {"type":"APPLY","tool":"read_file","content":"app.py"}\n'
            'ACTION {"type":"APPLY","tool":"apply_patch","content":"{\\"ops\\":[...]}"}\n'
            'ACTION {"type":"CHECK","tool":"run_unit_tests","content":""}\n'
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
                event_logger=event_logger,
            )
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
