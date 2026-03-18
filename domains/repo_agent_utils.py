from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


TEST_FILE_RE = re.compile(r"(^tests?/|/tests?/)")
TRACEBACK_FILE_RE = re.compile(r'File "([^"]+)", line (\d+)')


def create_workspace(task: Any, tmp_root: Path, *, deterministic: bool = False) -> Path:
    fixture_ref = str(getattr(task, "meta", {}).get("fixture_dir", "")).strip()
    suffix = "det" if deterministic else uuid.uuid4().hex[:8]
    workspace = tmp_root / f"{getattr(task, 'task_id', 'task')}_{suffix}"
    workspace.parent.mkdir(parents=True, exist_ok=True)
    if deterministic and workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    if fixture_ref:
        shutil.copytree(Path(fixture_ref), workspace)
        return workspace
    workspace.mkdir(parents=True, exist_ok=True)
    prompt = str(getattr(task, "prompt", "")).strip() or "No task prompt provided."
    (workspace / "TASK.md").write_text(prompt + "\n", encoding="utf-8")
    return workspace


def list_workspace_files(workspace: Path) -> List[str]:
    files: List[str] = []
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        relpath = str(path.relative_to(workspace)).replace("\\", "/")
        if "__pycache__" in relpath or relpath.endswith((".pyc", ".pyo")):
            continue
        files.append(relpath)
    return sorted(files)


def infer_primary_file(
    files: Sequence[str],
    *,
    preferred_file: str = "",
    candidate_source_files: Sequence[str] | None = None,
) -> str:
    if preferred_file and preferred_file in files:
        return preferred_file
    for name in candidate_source_files or []:
        if name in files:
            return name
    for name in files:
        if name.endswith(".py") and not TEST_FILE_RE.search(name):
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


def _latest_payload(state: Any, tool_name: str) -> Dict[str, Any]:
    for record in reversed(list(getattr(state, "tool_payloads", []))):
        if isinstance(record, dict) and record.get("tool") == tool_name:
            payload = record.get("payload", {})
            if isinstance(payload, dict):
                return payload
    return {}


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
            "test_output": output,
            "evidence": [f"test return code {proc.returncode}"],
            "resolved_obligations": ["verify tests"] if passed else [],
            "obligations": [] if passed else ["localize failure", "draft patch"],
        },
    }


def _repo_text_bundle(workspace: Path) -> tuple[Dict[str, str], Dict[str, str]]:
    source_files: Dict[str, str] = {}
    test_files: Dict[str, str] = {}
    for relpath in list_workspace_files(workspace):
        text = read_workspace_file(workspace, relpath)
        if TEST_FILE_RE.search(relpath.replace("\\", "/")):
            test_files[relpath] = text
        else:
            source_files[relpath] = text
    return source_files, test_files


def _literal_value(node: ast.AST) -> Any:
    return ast.literal_eval(node)


def _extract_unittest_cases(test_path: str, text: str) -> List[Dict[str, Any]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    imports: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imports[alias.asname or alias.name] = node.module
    cases: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "assertEqual":
            continue
        if len(node.args) < 2:
            continue
        call = node.args[0]
        expected_node = node.args[1]
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        try:
            args = [_literal_value(arg) for arg in call.args]
            expected = _literal_value(expected_node)
        except Exception:
            continue
        function_name = call.func.id
        module_name = imports.get(function_name, "")
        source_file = f"{module_name.replace('.', '/')}.py" if module_name else ""
        cases.append(
            {
                "test_file": test_path,
                "function_name": function_name,
                "module_name": module_name,
                "source_file": source_file,
                "args": args,
                "expected": expected,
            }
        )
    return cases


def inspect_python_tests(workspace: Path) -> Dict[str, Any]:
    files = list_workspace_files(workspace)
    test_files = [name for name in files if TEST_FILE_RE.search(name.replace("\\", "/"))]
    cases: List[Dict[str, Any]] = []
    source_candidates: List[str] = []
    symbols: List[str] = []
    for relpath in test_files:
        extracted = _extract_unittest_cases(relpath, read_workspace_file(workspace, relpath))
        cases.extend(extracted)
        for case in extracted:
            if case["source_file"]:
                source_candidates.append(str(case["source_file"]))
            if case["function_name"]:
                symbols.append(str(case["function_name"]))
    dedup_source = list(dict.fromkeys(source_candidates))
    dedup_symbols = list(dict.fromkeys(symbols))
    summary = {
        "test_files": test_files,
        "cases": cases,
        "candidate_source_files": dedup_source,
        "symbols": dedup_symbols,
    }
    return summary


def inspect_tests_tool(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    summary = inspect_python_tests(workspace)
    primary_file = infer_primary_file(
        list_workspace_files(workspace),
        preferred_file=str(state.metadata.get("primary_file", "")),
        candidate_source_files=summary["candidate_source_files"],
    )
    text_lines = [f"tests: {', '.join(summary['test_files']) or 'none'}"]
    if summary["symbols"]:
        text_lines.append(f"symbols: {', '.join(summary['symbols'])}")
    if summary["candidate_source_files"]:
        text_lines.append(f"candidate sources: {', '.join(summary['candidate_source_files'])}")
    return {
        "ok": True,
        "result": "\n".join(text_lines),
        "goal_progress": 0.22,
        "payload": {
            "test_summary": summary,
            "evidence": summary["test_files"][:3] + [f"targets {symbol}" for symbol in summary["symbols"][:2]],
            "obligations": ["run tests", "localize failure", "draft patch"],
            "resolved_obligations": ["inspect tests"],
            "state_metadata": {
                "primary_file": primary_file,
                "candidate_source_files": summary["candidate_source_files"],
                "test_symbols": summary["symbols"],
            },
        },
    }


def summarize_test_failures(output: str) -> Dict[str, Any]:
    line_refs: List[str] = []
    suspected_files: List[str] = []
    for path, line_no in TRACEBACK_FILE_RE.findall(output or ""):
        relpath = str(path).replace("\\", "/")
        line_refs.append(f"{relpath}:{line_no}")
        if relpath.endswith(".py"):
            suspected_files.append(relpath)
    return {
        "line_refs": list(dict.fromkeys(line_refs)),
        "suspected_files": list(dict.fromkeys(suspected_files)),
    }


def localize_failure_tool(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    summary = _latest_payload(state, "inspect_tests").get("test_summary") or inspect_python_tests(workspace)
    last_test_output = ""
    for record in reversed(list(getattr(state, "tool_history", []))):
        if isinstance(record, dict) and record.get("tool") == "run_unit_tests":
            result = record.get("result", {})
            if isinstance(result, dict):
                last_test_output = str(result.get("result_text", "") or result.get("result", ""))
                break
    failure = summarize_test_failures(last_test_output)
    candidate_files = [name for name in failure["suspected_files"] if not TEST_FILE_RE.search(name)]
    if not candidate_files:
        candidate_files = list(summary.get("candidate_source_files", []))
    primary_file = infer_primary_file(
        list_workspace_files(workspace),
        preferred_file=str(state.metadata.get("primary_file", "")),
        candidate_source_files=candidate_files,
    )
    evidence = []
    if failure["line_refs"]:
        evidence.extend(failure["line_refs"][:3])
    if summary.get("symbols"):
        evidence.append(f"tests call {', '.join(summary['symbols'][:2])}")
    payload = {
        "candidate_source_files": candidate_files,
        "primary_file": primary_file,
        "suspected_symbols": list(summary.get("symbols", [])),
        "failure_lines": failure["line_refs"],
    }
    return {
        "ok": True,
        "result": json.dumps(payload, ensure_ascii=True),
        "goal_progress": 0.28,
        "payload": {
            "localization": payload,
            "evidence": evidence or [f"primary file {primary_file}"],
            "obligations": ["inspect source", "draft patch", "verify tests"],
            "resolved_obligations": ["localize failure"],
            "state_metadata": {"primary_file": primary_file},
        },
    }


def _apply_ops_to_text(text: str, ops: Sequence[Dict[str, str]]) -> str | None:
    updated = text
    for op in ops:
        search = str(op.get("search", ""))
        replace = str(op.get("replace", ""))
        if search not in updated:
            return None
        updated = updated.replace(search, replace, 1)
    return updated


def _score_cases_with_source(module_text: str, cases: Sequence[Dict[str, Any]]) -> tuple[int, int]:
    if not cases:
        return (0, 0)
    namespace: Dict[str, Any] = {}
    try:
        exec(module_text, namespace, namespace)
    except Exception:
        return (0, len(cases))
    passed = 0
    for case in cases:
        fn = namespace.get(str(case.get("function_name", "")))
        if not callable(fn):
            continue
        try:
            actual = fn(*list(case.get("args", [])))
        except Exception:
            continue
        if actual == case.get("expected"):
            passed += 1
    return (passed, len(cases))


def _generate_case_driven_candidates(relpath: str, text: str, cases: Sequence[Dict[str, Any]], prompt: str) -> List[Dict[str, Any]]:
    del prompt
    candidates: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(search: str, replace: str, reason: str, *, generator: str = "case_driven") -> None:
        key = (search, replace)
        if not search or search not in text or key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "ops": [{"path": relpath, "search": search, "replace": replace}],
                "evidence": [reason],
                "provenance": [reason],
                "generator": generator,
                "source_file": relpath,
            }
        )

    for match in re.finditer(r"return\s+([A-Za-z_][A-Za-z0-9_]*)\s*-\s*([A-Za-z_][A-Za-z0-9_]*)", text):
        add(match.group(0), f"return {match.group(1)} + {match.group(2)}", f"flip arithmetic operator in {relpath}")
    for match in re.finditer(r"return\s+([A-Za-z_][A-Za-z0-9_]*)\s*\+\s*([A-Za-z_][A-Za-z0-9_]*)", text):
        add(match.group(0), f"return {match.group(1)} - {match.group(2)}", f"flip arithmetic operator in {relpath}")
    if ">= 0" in text:
        add(">= 0", "> 0", f"tighten positive-threshold comparison in {relpath}")
    if "<= 0" in text:
        add("<= 0", "< 0", f"tighten negative-threshold comparison in {relpath}")
    if "len(items) - 1" in text:
        add("len(items) - 1", "len(items)", f"fix off-by-one length expression in {relpath}")
    for match in re.finditer(
        r"return\s+([A-Za-z_][A-Za-z0-9_]*)\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
        text,
    ):
        fn_name, first_arg, second_arg = match.groups()
        add(match.group(0), f"return {fn_name}({second_arg}, {first_arg})", f"swap argument order in {relpath}")
    for match in re.finditer(
        r'return\s+f"\{([A-Za-z_][A-Za-z0-9_]*)\}\s+\{([A-Za-z_][A-Za-z0-9_]*)\}"',
        text,
    ):
        first_arg, second_arg = match.groups()
        add(match.group(0), f'return f"{{{second_arg}}} {{{first_arg}}}"', f"swap formatted name order in {relpath}", generator="case_driven_fstring")

    expected_strings = [str(case["expected"]) for case in cases if isinstance(case.get("expected"), str)]
    preferred_separator = "-" if any("-" in value for value in expected_strings) else "_" if any("_" in value for value in expected_strings) else ""
    if preferred_separator:
        for match in re.finditer(
            r'return\s+([A-Za-z_][A-Za-z0-9_]*)\.strip\(\)\.lower\(\)\.replace\(" ",\s*"[_-]"\)',
            text,
        ):
            variable = match.group(1)
            replacement = f'return "{preferred_separator}".join({variable}.strip().lower().split())'
            add(match.group(0), replacement, f"normalize whitespace with '{preferred_separator}' separator in {relpath}")
        if '.replace(" ", "_")' in text and preferred_separator == "-":
            add('.replace(" ", "_")', '.replace(" ", "-")', f"switch separator in {relpath}")

    return candidates


def _pattern_patch_candidates(source_files: Dict[str, str], test_files: Dict[str, str], prompt: str, preferred_file: str = "") -> List[Dict[str, Any]]:
    prompt_lower = prompt.lower()
    test_text = "\n".join(test_files.values()).lower()
    ordered_sources = list(source_files.items())
    if preferred_file and preferred_file in source_files:
        ordered_sources = [(preferred_file, source_files[preferred_file])] + [(path, text) for path, text in ordered_sources if path != preferred_file]

    candidates: List[Dict[str, Any]] = []
    for relpath, text in ordered_sources:
        if "return a - b" in text and any(token in (prompt_lower + test_text) for token in ["add", "sum", "arithmetic"]):
            candidates.append(
                {
                    "ops": [{"path": relpath, "search": "    return a - b\n", "replace": "    return a + b\n"}],
                    "evidence": [f"detected subtraction bug in {relpath}"],
                    "provenance": ["pattern: subtraction-to-addition"],
                    "generator": "pattern",
                    "source_file": relpath,
                }
            )
        if '.replace(" ", "_")' in text and any(token in (prompt_lower + test_text) for token in ["slug", "hyphen", "ada-lovelace"]):
            candidates.append(
                {
                    "ops": [{"path": relpath, "search": '    return value.strip().lower().replace(" ", "_")\n', "replace": '    return "-".join(value.strip().lower().split())\n'}],
                    "evidence": [f"detected slug formatting bug in {relpath}"],
                    "provenance": ["pattern: slug-separator-normalization"],
                    "generator": "pattern",
                    "source_file": relpath,
                }
            )
        if ">= 0" in text and any(token in (prompt_lower + test_text) for token in ["positive", "exclude zero", "only positive"]):
            candidates.append(
                {
                    "ops": [{"path": relpath, "search": ">= 0", "replace": "> 0"}],
                    "evidence": [f"detected inclusive-threshold bug in {relpath}"],
                    "provenance": ["pattern: inclusive-threshold"],
                    "generator": "pattern",
                    "source_file": relpath,
                }
            )
        if "len(items) - 1" in text and "count" in (prompt_lower + test_text):
            candidates.append(
                {
                    "ops": [{"path": relpath, "search": "len(items) - 1", "replace": "len(items)"}],
                    "evidence": [f"detected off-by-one count bug in {relpath}"],
                    "provenance": ["pattern: off-by-one-count"],
                    "generator": "pattern",
                    "source_file": relpath,
                }
            )
    return candidates


def draft_patch_tool(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    source_files, test_files = _repo_text_bundle(workspace)
    prompt = arg.strip() or str(getattr(state, "problem_text", ""))

    test_summary = _latest_payload(state, "inspect_tests").get("test_summary") or inspect_python_tests(workspace)
    localization = _latest_payload(state, "localize_failure").get("localization", {})
    candidate_files = list(localization.get("candidate_source_files", [])) or list(test_summary.get("candidate_source_files", []))
    preferred_file = infer_primary_file(
        list_workspace_files(workspace),
        preferred_file=str(state.metadata.get("primary_file", "")),
        candidate_source_files=candidate_files,
    )
    ordered_files = [preferred_file] + [name for name in candidate_files if name != preferred_file]
    ordered_files.extend([name for name in source_files if name not in ordered_files])

    cases_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for case in list(test_summary.get("cases", [])):
        source_file = str(case.get("source_file", "")).strip()
        if source_file:
            cases_by_source.setdefault(source_file, []).append(case)

    ranked_candidates: List[Dict[str, Any]] = []
    failed_candidates: List[Dict[str, Any]] = []

    for relpath in ordered_files:
        source_text = source_files.get(relpath)
        if not source_text:
            continue
        relevant_cases = cases_by_source.get(relpath, [])
        for candidate in _generate_case_driven_candidates(relpath, source_text, relevant_cases, prompt):
            updated = _apply_ops_to_text(source_text, candidate["ops"])
            if updated is None:
                continue
            passed, total = _score_cases_with_source(updated, relevant_cases)
            fit = float(passed) / float(max(1, total)) if total > 0 else 0.0
            score = fit + (0.05 if relpath == preferred_file else 0.0)
            candidate["validated_cases"] = {"passed": passed, "total": total}
            candidate["score"] = score
            if total > 0 and passed == total:
                ranked_candidates.append(candidate)
            else:
                failed_candidates.append(candidate)

    for candidate in _pattern_patch_candidates(source_files, test_files, prompt, preferred_file=preferred_file):
        relpath = str(candidate.get("source_file", ""))
        source_text = source_files.get(relpath, "")
        updated = _apply_ops_to_text(source_text, candidate["ops"])
        if updated is None:
            continue
        relevant_cases = cases_by_source.get(relpath, [])
        passed, total = _score_cases_with_source(updated, relevant_cases)
        fit = float(passed) / float(max(1, total)) if total > 0 else 0.25
        score = fit + 0.02
        candidate["validated_cases"] = {"passed": passed, "total": total}
        candidate["score"] = score
        if total == 0 or passed == total:
            ranked_candidates.append(candidate)
        else:
            failed_candidates.append(candidate)

    ranked_candidates.sort(
        key=lambda item: (
            float(item.get("score", 0.0)),
            int(item.get("validated_cases", {}).get("passed", 0)),
            1 if str(item.get("source_file", "")) == preferred_file else 0,
        ),
        reverse=True,
    )
    failed_candidates.sort(
        key=lambda item: (
            float(item.get("score", 0.0)),
            int(item.get("validated_cases", {}).get("passed", 0)),
        ),
        reverse=True,
    )

    if ranked_candidates:
        chosen = ranked_candidates[0]
        ops = chosen["ops"]
        evidence = list(chosen.get("evidence", []))
        validated = dict(chosen.get("validated_cases", {}))
        return {
            "ok": True,
            "result": json.dumps({"ops": ops}, ensure_ascii=True),
            "goal_progress": 0.62 if int(validated.get("total", 0)) > 0 else 0.55,
            "risk": 0.08 if int(validated.get("passed", 0)) == int(validated.get("total", 0)) and int(validated.get("total", 0)) > 0 else 0.15,
            "payload": {
                "patch_ops": ops,
                "evidence": evidence + ([f"validated against {validated.get('passed', 0)}/{validated.get('total', 0)} extracted tests"] if int(validated.get("total", 0)) > 0 else []),
                "obligations": ["apply patch", "verify tests"],
                "resolved_obligations": ["inspect source", "inspect tests"],
                "suggested_tools": ["apply_patch", "run_unit_tests"],
                "patch_candidates": [
                    {
                        "path": str(item["ops"][0]["path"]),
                        "score": round(float(item.get("score", 0.0)), 4),
                        "provenance": list(item.get("provenance", [])),
                        "validated_cases": dict(item.get("validated_cases", {})),
                    }
                    for item in ranked_candidates[:5]
                ],
                "failed_patch_candidates": [
                    {
                        "path": str(item["ops"][0]["path"]),
                        "score": round(float(item.get("score", 0.0)), 4),
                        "provenance": list(item.get("provenance", [])),
                        "validated_cases": dict(item.get("validated_cases", {})),
                    }
                    for item in failed_candidates[:5]
                ],
                "state_metadata": {
                    "primary_file": ops[0]["path"],
                    "patch_candidate_count": len(ranked_candidates),
                    "failed_patch_attempt_count": len(failed_candidates),
                },
            },
        }

    return {
        "ok": False,
        "result": "could not draft a patch from repository context",
        "goal_progress": 0.0,
        "risk": 0.7,
        "payload": {
            "obligations": ["inspect source", "inspect tests", "localize failure"],
            "failed_patch_candidates": [
                {
                    "path": str(item["ops"][0]["path"]),
                    "score": round(float(item.get("score", 0.0)), 4),
                    "provenance": list(item.get("provenance", [])),
                    "validated_cases": dict(item.get("validated_cases", {})),
                }
                for item in failed_candidates[:5]
            ],
        },
    }
