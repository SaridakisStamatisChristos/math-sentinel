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
    for key in ("fixture_dir", "workspace_dir", "attachments_dir", "attachment_dir", "workspace_archive"):
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


def load_manifest_suite(path_like: str) -> BenchmarkSuite:
    path = Path(path_like).resolve()
    payload = _read_manifest_payload(path)
    raw_cases = payload.get("cases", payload.get("tasks", []))
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"manifest {path} does not contain any cases")
    backend = str(payload.get("backend", "")).strip() or _infer_backend(raw_cases[0])
    name = str(payload.get("name", path.stem)).strip() or path.stem
    description = str(payload.get("description", f"Manifest benchmark imported from {path.name}")).strip()
    tier = str(payload.get("tier", payload.get("metadata", {}).get("tier", "official"))).strip() or "official"
    metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {}
    metadata["manifest_path"] = str(path)
    metadata["source_format"] = str(payload.get("source_format", path.suffix.lower().lstrip("."))).strip() or "json"
    cases = [_task_from_case(case, manifest_dir=path.parent, suite_name=name, suite_tier=tier, suite_backend=backend) for case in raw_cases]
    return BenchmarkSuite(name=name, backend=backend, description=description, tier=tier, cases=cases, metadata=metadata)


def manifest_suite_spec(path_like: str) -> str:
    return f"manifest:{path_like}"
