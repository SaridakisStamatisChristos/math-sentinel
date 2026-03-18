from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def create_workspace(task: Any, tmp_root: Path) -> Path:
    fixture_ref = str(getattr(task, "meta", {}).get("fixture_dir", "")).strip()
    workspace = tmp_root / f"{getattr(task, 'task_id', 'task')}_{uuid.uuid4().hex[:8]}"
    workspace.parent.mkdir(parents=True, exist_ok=True)
    if fixture_ref:
        shutil.copytree(Path(fixture_ref), workspace)
        return workspace
    workspace.mkdir(parents=True, exist_ok=True)
    prompt = str(getattr(task, "prompt", "")).strip() or "No task prompt provided."
    (workspace / "TASK.md").write_text(prompt + "\n", encoding="utf-8")
    return workspace


def list_workspace_files(workspace: Path) -> List[str]:
    return sorted(
        str(path.relative_to(workspace)).replace("\\", "/")
        for path in workspace.rglob("*")
        if path.is_file()
    )


def infer_primary_file(files: Sequence[str]) -> str:
    for name in files:
        if name.endswith(".py") and not name.startswith("tests/"):
            return name
    return files[0] if files else ""


def read_workspace_file(workspace: Path, relpath: str) -> str:
    return (workspace / relpath).read_text(encoding="utf-8")


def parse_patch_ops(text: str) -> List[Dict[str, str]]:
    normalized = text.strip()
    if not normalized:
        return []
    payload = json.loads(normalized)
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


def _latest_patch_ops_from_state(state: Any) -> List[Dict[str, str]]:
    for record in reversed(list(getattr(state, "tool_payloads", []))):
        payload = record.get("payload", {}) if isinstance(record, dict) else {}
        ops = payload.get("patch_ops", [])
        if isinstance(ops, list) and ops:
            return [
                {
                    "path": str(item.get("path", "")),
                    "search": str(item.get("search", "")),
                    "replace": str(item.get("replace", "")),
                }
                for item in ops
                if isinstance(item, dict) and item.get("path")
            ]
    return []


def apply_patch_ops(workspace: Path, ops: List[Dict[str, str]]) -> Dict[str, Any]:
    changed_files: List[str] = []
    backups: List[Dict[str, str]] = []
    for op in ops:
        path = workspace / op["path"]
        original = path.read_text(encoding="utf-8")
        if op["search"] not in original:
            return {"ok": False, "result": f"search text not found in {op['path']}", "risk": 0.8}
        updated = original.replace(op["search"], op["replace"], 1)
        path.write_text(updated, encoding="utf-8")
        changed_files.append(op["path"])
        backups.append({"path": op["path"], "content": original})
    return {
        "ok": True,
        "result": f"patched files: {', '.join(changed_files)}",
        "goal_progress": 0.7,
        "payload": {
            "changed_files": changed_files,
            "patch_ops": ops,
            "backups": backups,
            "obligations": ["verify tests"],
            "resolved_obligations": ["draft patch"],
        },
    }


def apply_patch_tool(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    ops = parse_patch_ops(arg) if arg.strip() else _latest_patch_ops_from_state(state)
    if not ops:
        return {"ok": False, "result": "no patch operations available", "risk": 0.8}
    return apply_patch_ops(workspace, ops)


def rollback_patch_tool(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    backups: List[Dict[str, str]] = []
    for record in reversed(list(getattr(state, "tool_payloads", []))):
        payload = record.get("payload", {}) if isinstance(record, dict) else {}
        backup_items = payload.get("backups", [])
        if isinstance(backup_items, list) and backup_items:
            backups = [item for item in backup_items if isinstance(item, dict) and item.get("path")]
            break
    if not backups:
        return {"ok": False, "result": "no backups available for rollback", "risk": 0.7}
    restored: List[str] = []
    for item in backups:
        (workspace / str(item["path"])).write_text(str(item.get("content", "")), encoding="utf-8")
        restored.append(str(item["path"]))
    return {
        "ok": True,
        "result": f"restored files: {', '.join(restored)}",
        "goal_progress": 0.2,
        "payload": {"changed_files": restored, "resolved_obligations": ["verify tests"]},
    }


def run_unit_tests_tool(arg: str, state: Any = None) -> Dict[str, Any]:
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
        "payload": {
            "command": resolved,
            "returncode": proc.returncode,
            "evidence": [f"test return code {proc.returncode}"],
            "resolved_obligations": ["verify tests"] if passed else [],
        },
    }


def _repo_text_bundle(workspace: Path) -> tuple[Dict[str, str], Dict[str, str]]:
    source_files: Dict[str, str] = {}
    test_files: Dict[str, str] = {}
    for relpath in list_workspace_files(workspace):
        text = read_workspace_file(workspace, relpath)
        if relpath.startswith("tests/") or "/tests/" in relpath.replace("\\", "/"):
            test_files[relpath] = text
        else:
            source_files[relpath] = text
    return source_files, test_files


def _pattern_patch_ops(source_files: Dict[str, str], test_files: Dict[str, str], prompt: str, preferred_file: str = "") -> tuple[List[Dict[str, str]], List[str]]:
    prompt_lower = prompt.lower()
    test_text = "\n".join(test_files.values()).lower()
    ordered_sources = list(source_files.items())
    if preferred_file and preferred_file in source_files:
        ordered_sources = [(preferred_file, source_files[preferred_file])] + [(path, text) for path, text in ordered_sources if path != preferred_file]

    for relpath, text in ordered_sources:
        if "return a - b" in text and any(token in (prompt_lower + test_text) for token in ["add", "sum", "arithmetic"]):
            return ([{"path": relpath, "search": "    return a - b\n", "replace": "    return a + b\n"}], [f"detected subtraction bug in {relpath}"])
        if '.replace(" ", "_")' in text and any(token in (prompt_lower + test_text) for token in ["slug", "hyphen", "ada-lovelace"]):
            return ([{"path": relpath, "search": '    return value.strip().lower().replace(" ", "_")\n', "replace": '    return "-".join(value.strip().lower().split())\n'}], [f"detected slug formatting bug in {relpath}"])
        if ">= 0" in text and any(token in (prompt_lower + test_text) for token in ["positive", "exclude zero", "only positive"]):
            return ([{"path": relpath, "search": ">= 0", "replace": "> 0"}], [f"detected inclusive-threshold bug in {relpath}"])
        if "len(items) - 1" in text and "count" in (prompt_lower + test_text):
            return ([{"path": relpath, "search": "len(items) - 1", "replace": "len(items)"}], [f"detected off-by-one count bug in {relpath}"])
    return ([], [])


def draft_patch_tool(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    source_files, test_files = _repo_text_bundle(workspace)
    preferred_file = str(state.metadata.get("primary_file", ""))
    prompt = arg.strip() or str(getattr(state, "problem_text", ""))
    ops, evidence = _pattern_patch_ops(source_files, test_files, prompt, preferred_file=preferred_file)
    if not ops:
        return {
            "ok": False,
            "result": "could not draft a patch from repository context",
            "goal_progress": 0.0,
            "risk": 0.7,
            "payload": {"evidence": evidence, "obligations": ["inspect source and tests"]},
        }
    return {
        "ok": True,
        "result": json.dumps({"ops": ops}, ensure_ascii=True),
        "goal_progress": 0.55,
        "risk": 0.1,
        "payload": {
            "patch_ops": ops,
            "evidence": evidence,
            "obligations": ["draft patch", "apply patch", "verify tests"],
            "resolved_obligations": ["inspect source", "inspect tests"],
            "suggested_tools": ["apply_patch", "run_unit_tests"],
        },
    }
