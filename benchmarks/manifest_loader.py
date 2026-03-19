from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from engine.task import ReasoningTask

from .base import BenchmarkSuite


def _read_manifest_payload(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".jsonl":
        cases: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                cases.append(json.loads(text))
        return {"cases": cases}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {"cases": payload}
    if not isinstance(payload, dict):
        raise ValueError(f"unsupported manifest payload in {path}")
    return payload


def _resolve_metadata_paths(meta: Dict[str, Any], manifest_dir: Path) -> Dict[str, Any]:
    resolved = dict(meta)
    for key in ("fixture_dir", "workspace_dir", "attachments_dir", "attachment_dir", "workspace_archive", "repo_cache_root"):
        value = resolved.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (manifest_dir / candidate).resolve()
        resolved[key] = str(candidate)
    return resolved


def _infer_backend(raw_case: Dict[str, Any]) -> str:
    backend = str(raw_case.get("backend", "")).strip()
    if backend:
        return backend
    domain = str(raw_case.get("domain", "")).strip().lower()
    meta = raw_case.get("meta", raw_case.get("metadata", {}))
    family = str(meta.get("family", "")).strip().lower() if isinstance(meta, dict) else ""
    combined = f"{domain} {family}"
    if "swebench" in combined:
        return "swebench_ops"
    if "gaia" in combined:
        return "gaia_ops"
    if any(token in combined for token in ["arith", "linear_equation", "fraction", "derivative", "integral", "factor", "modular", "gcd", "lcm"]):
        return "math"
    if "project_plan" in combined or "shopping_plan" in combined or "day_plan" in combined:
        return "planning_ops"
    if "code" in combined or "repo_patch" in combined:
        return "code_ops"
    raise ValueError("could not infer backend from manifest case; add a suite-level backend or case backend field")


def _task_from_case(raw_case: Dict[str, Any], *, manifest_dir: Path, suite_name: str, suite_tier: str, suite_backend: str) -> ReasoningTask:
    meta = raw_case.get("meta", raw_case.get("metadata", {}))
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        raise ValueError("manifest case meta/metadata must be a dictionary")
    meta = _resolve_metadata_paths(meta, manifest_dir)
    meta.setdefault("benchmark_suite", suite_name)
    meta.setdefault("benchmark_tier", suite_tier)
    meta.setdefault("holdout_group", suite_name)
    meta.setdefault("source", "public_benchmark")
    meta.setdefault("fixture_role", "benchmark")
    backend = str(raw_case.get("backend", "")).strip() or suite_backend
    if backend:
        meta.setdefault("manifest_backend", backend)
    task_id = str(raw_case.get("task_id", raw_case.get("id", raw_case.get("instance_id", "")))).strip()
    if not task_id:
        raise ValueError("manifest case is missing task_id")
    domain = str(raw_case.get("domain", meta.get("family", ""))).strip()
    if not domain:
        raise ValueError(f"manifest case {task_id} is missing domain/family")
    prompt = str(raw_case.get("prompt", raw_case.get("problem_statement", raw_case.get("question", "")))).strip()
    if not prompt:
        raise ValueError(f"manifest case {task_id} is missing prompt/problem_statement/question")
    answer = str(
        raw_case.get(
            "answer",
            raw_case.get("expected_answer", raw_case.get("final_answer", "patched_and_verified" if backend == "swebench_ops" else "")),
        )
    ).strip()
    goal = str(
        raw_case.get(
            "goal",
            "Patch the repository so the tests pass" if backend == "swebench_ops" else "Return the shortest correct final answer",
        )
    ).strip()
    return ReasoningTask(task_id=task_id, domain=domain, prompt=prompt, answer=answer, goal=goal, meta=meta)


def _case_problems(
    raw_case: Dict[str, Any],
    *,
    manifest_dir: Path,
    suite_backend: str,
    strict_materialization: bool,
) -> tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    meta = raw_case.get("meta", raw_case.get("metadata", {}))
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        return (["manifest case meta/metadata must be a dictionary"], warnings)
    meta = _resolve_metadata_paths(meta, manifest_dir)

    task_id = str(raw_case.get("task_id", raw_case.get("id", raw_case.get("instance_id", "")))).strip()
    if not task_id:
        errors.append("manifest case is missing task_id")
    domain = str(raw_case.get("domain", meta.get("family", ""))).strip()
    if not domain:
        errors.append(f"manifest case {task_id or '<unknown>'} is missing domain/family")
    prompt = str(raw_case.get("prompt", raw_case.get("problem_statement", raw_case.get("question", "")))).strip()
    if not prompt:
        errors.append(f"manifest case {task_id or '<unknown>'} is missing prompt/problem_statement/question")
    answer = str(raw_case.get("answer", raw_case.get("expected_answer", raw_case.get("final_answer", "")))).strip()
    if not answer:
        warnings.append(f"manifest case {task_id or '<unknown>'} is missing answer/expected_answer/final_answer")

    for key in ("fixture_dir", "workspace_dir", "attachments_dir", "attachment_dir", "workspace_archive"):
        value = meta.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        target = Path(value)
        if not target.exists():
            message = f"manifest case {task_id or '<unknown>'} references missing {key}: {target}"
            if strict_materialization:
                errors.append(message)
            else:
                warnings.append(message)
            continue
        if key.endswith("_dir") and not target.is_dir():
            errors.append(f"manifest case {task_id or '<unknown>'} expected directory for {key}: {target}")
    if suite_backend == "swebench_ops":
        has_materialized_repo = any(str(meta.get(key, "")).strip() for key in ("fixture_dir", "workspace_dir"))
        has_repo_source = all(str(meta.get(key, "")).strip() for key in ("repo", "base_commit"))
        if not has_materialized_repo and not has_repo_source:
            warnings.append(
                f"manifest case {task_id or '<unknown>'} has no fixture_dir/workspace_dir or repo/base_commit for repo materialization"
            )
    if suite_backend == "gaia_ops" and not any(str(meta.get(key, "")).strip() for key in ("fixture_dir", "workspace_dir", "attachments_dir", "attachment_dir")):
        warnings.append(f"manifest case {task_id or '<unknown>'} has no materialized evidence directory")
    return errors, warnings


def lint_manifest_suite(path_like: str, *, strict_materialization: bool = True) -> Dict[str, Any]:
    path = Path(path_like).resolve()
    payload = _read_manifest_payload(path)
    raw_cases = payload.get("cases", payload.get("tasks", []))
    if not isinstance(raw_cases, list) or not raw_cases:
        return {
            "valid": False,
            "name": str(payload.get("name", path.stem)).strip() or path.stem,
            "backend": str(payload.get("backend", "")).strip(),
            "case_count": 0,
            "errors": [f"manifest {path} does not contain any cases"],
            "warnings": [],
            "manifest_path": str(path),
        }

    suite_name = str(payload.get("name", path.stem)).strip() or path.stem
    backend = str(payload.get("backend", "")).strip()
    errors: List[str] = []
    warnings: List[str] = []
    if not backend:
        try:
            backend = _infer_backend(raw_cases[0])
        except Exception as exc:
            errors.append(str(exc))
    seen_task_ids: set[str] = set()
    duplicate_ids: List[str] = []
    for raw_case in raw_cases:
        task_id = str(raw_case.get("task_id", raw_case.get("id", raw_case.get("instance_id", "")))).strip()
        if task_id:
            if task_id in seen_task_ids:
                duplicate_ids.append(task_id)
            seen_task_ids.add(task_id)
        case_errors, case_warnings = _case_problems(
            raw_case,
            manifest_dir=path.parent,
            suite_backend=backend,
            strict_materialization=strict_materialization,
        )
        errors.extend(case_errors)
        warnings.extend(case_warnings)
    if duplicate_ids:
        errors.append(f"manifest contains duplicate task_id values: {', '.join(sorted(set(duplicate_ids)))}")
    return {
        "valid": not errors,
        "name": suite_name,
        "backend": backend,
        "case_count": len(raw_cases),
        "errors": errors,
        "warnings": warnings,
        "manifest_path": str(path),
    }


def load_manifest_suite(path_like: str, *, max_cases: int | None = None) -> BenchmarkSuite:
    path = Path(path_like).resolve()
    lint = lint_manifest_suite(str(path), strict_materialization=True)
    if not lint["valid"]:
        raise ValueError("; ".join(str(item) for item in lint["errors"]))
    payload = _read_manifest_payload(path)
    raw_cases = payload.get("cases", payload.get("tasks", []))
    if max_cases is not None:
        raw_cases = raw_cases[: max(0, int(max_cases))]
    if not raw_cases:
        raise ValueError(f"manifest {path} does not contain any cases after applying max_cases")
    backend = str(payload.get("backend", "")).strip() or str(lint["backend"]).strip() or _infer_backend(raw_cases[0])
    name = str(payload.get("name", path.stem)).strip() or str(lint["name"]).strip() or path.stem
    description = str(payload.get("description", f"Manifest benchmark imported from {path.name}")).strip()
    tier = str(payload.get("tier", payload.get("metadata", {}).get("tier", "official"))).strip() or "official"
    metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {}
    metadata["manifest_path"] = str(path)
    metadata["source_format"] = str(payload.get("source_format", path.suffix.lower().lstrip("."))).strip() or "json"
    metadata["manifest_warnings"] = list(lint.get("warnings", []))
    cases = [_task_from_case(case, manifest_dir=path.parent, suite_name=name, suite_tier=tier, suite_backend=backend) for case in raw_cases]
    return BenchmarkSuite(name=name, backend=backend, description=description, tier=tier, cases=cases, metadata=metadata)


def manifest_suite_spec(path_like: str) -> str:
    return f"manifest:{path_like}"
