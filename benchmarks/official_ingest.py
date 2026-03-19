from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .manifest_loader import lint_manifest_suite


def _first_text(record: Dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _metadata_block(record: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("meta", "metadata"):
        value = record.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _read_records(path_like: str) -> List[Dict[str, Any]]:
    path = Path(path_like)
    if path.suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if isinstance(payload, dict):
                    records.append(payload)
        return records
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        inner = payload.get("records", payload.get("cases", payload.get("tasks", [])))
        if isinstance(inner, list):
            return [item for item in inner if isinstance(item, dict)]
    raise ValueError(f"unsupported record input: {path_like}")


def _relpath_for_manifest(target: Path, *, output_dir: Path) -> str:
    try:
        return str(target.resolve().relative_to(output_dir.resolve())).replace("\\", "/")
    except Exception:
        return str(target.resolve())


def _fixture_path(record: Dict[str, Any], fixtures_root: str) -> str:
    for key in (
        "fixture_dir",
        "workspace_dir",
        "workspace_path",
        "workspace",
        "attachments_dir",
        "attachment_dir",
        "artifact_dir",
    ):
        value = str(record.get(key, "")).strip()
        if value:
            return value
    root = Path(fixtures_root) if fixtures_root else None
    if root is None:
        return ""
    for key in ("task_id", "instance_id", "id", "question_id"):
        value = str(record.get(key, "")).strip()
        if value:
            candidate = root / value
            if candidate.exists():
                return str(candidate)
    return ""


def _write_manifest(output_path: str, payload: Dict[str, Any]) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return str(path)


def ingest_swebench_records(
    input_path: str,
    output_path: str,
    *,
    fixtures_root: str = "",
    suite_name: str = "swebench_verified_public_import",
    tier: str = "official",
    description: str = "Official-style SWE-bench manifest imported from local records.",
) -> str:
    records = _read_records(input_path)
    output_dir = Path(output_path).resolve().parent
    cases: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        task_id = _first_text(record, "task_id", "instance_id", "id", default=f"swebench_import_{idx}")
        prompt = _first_text(record, "prompt", "problem_statement", "issue", "problem", "statement")
        fixture_dir = _fixture_path(record, fixtures_root)
        meta = _metadata_block(record)
        meta.update(
            {
                "family": "swebench_patch",
                "benchmark_suite": suite_name,
                "benchmark_tier": tier,
                "holdout_group": suite_name,
                "source": "public_benchmark",
                "fixture_role": "benchmark",
            }
        )
        if fixture_dir:
            meta["fixture_dir"] = _relpath_for_manifest(Path(fixture_dir), output_dir=output_dir)
        if "test_command" in record:
            meta["test_command"] = record["test_command"]
        elif "test_cmd" in record:
            meta["test_command"] = record["test_cmd"]
        if "oracle_primary_file" in record:
            meta["oracle_primary_file"] = record["oracle_primary_file"]
        elif "primary_file" in record:
            meta["oracle_primary_file"] = record["primary_file"]
        if "oracle_patch" in record:
            meta["oracle_patch"] = record["oracle_patch"]
        elif "patch" in record:
            meta["oracle_patch"] = record["patch"]
        cases.append(
            {
                "task_id": task_id,
                "domain": "swebench_patch",
                "prompt": prompt,
                "answer": _first_text(record, "answer", "expected_answer", "final_answer", default="patched_and_verified") or "patched_and_verified",
                "goal": _first_text(record, "goal", default="Patch the repository so the tests pass"),
                "meta": meta,
            }
        )
    payload = {
        "name": suite_name,
        "backend": "swebench_ops",
        "description": description,
        "tier": tier,
        "source_format": "swebench_jsonl",
        "metadata": {"record_count": len(cases), "input_path": str(Path(input_path).resolve())},
        "cases": cases,
    }
    written = _write_manifest(output_path, payload)
    lint = lint_manifest_suite(written, strict_materialization=True)
    if not lint["valid"]:
        raise ValueError("; ".join(str(item) for item in lint["errors"]))
    return written


def _infer_gaia_domain(record: Dict[str, Any], fixture_dir: str) -> str:
    explicit = str(record.get("domain", record.get("family", ""))).strip()
    if explicit:
        return explicit
    combined = json.dumps(record, ensure_ascii=True).lower()
    combined += " " + fixture_dir.lower()
    if ".csv" in combined:
        return "gaia_csv_reasoning"
    if ".json" in combined:
        return "gaia_json_reasoning"
    return "gaia_json_reasoning"


def ingest_gaia_records(
    input_path: str,
    output_path: str,
    *,
    fixtures_root: str = "",
    suite_name: str = "gaia_public_import",
    tier: str = "official",
    description: str = "Official-style GAIA manifest imported from local records.",
) -> str:
    records = _read_records(input_path)
    output_dir = Path(output_path).resolve().parent
    cases: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        task_id = _first_text(record, "task_id", "question_id", "id", default=f"gaia_import_{idx}")
        prompt = _first_text(record, "prompt", "question", "problem_statement")
        fixture_dir = _fixture_path(record, fixtures_root)
        domain = _infer_gaia_domain(record, fixture_dir)
        meta = _metadata_block(record)
        meta.update(
            {
                "family": domain,
                "benchmark_suite": suite_name,
                "benchmark_tier": tier,
                "holdout_group": suite_name,
                "source": "public_benchmark",
                "fixture_role": "benchmark",
            }
        )
        if fixture_dir:
            meta["fixture_dir"] = _relpath_for_manifest(Path(fixture_dir), output_dir=output_dir)
        alias_map = {
            "oracle_tool": ("oracle_tool", "recommended_tool"),
            "oracle_input": ("oracle_input", "tool_input"),
            "oracle_evidence_file": ("oracle_evidence_file", "evidence_file", "primary_file"),
        }
        for target_key, source_keys in alias_map.items():
            value = _first_text(record, *source_keys)
            if value:
                meta[target_key] = value
        cases.append(
            {
                "task_id": task_id,
                "domain": domain,
                "prompt": prompt,
                "answer": _first_text(record, "answer", "final_answer", "expected_answer"),
                "goal": _first_text(record, "goal", default="Return the shortest correct final answer"),
                "meta": meta,
            }
        )
    payload = {
        "name": suite_name,
        "backend": "gaia_ops",
        "description": description,
        "tier": tier,
        "source_format": "gaia_jsonl",
        "metadata": {"record_count": len(cases), "input_path": str(Path(input_path).resolve())},
        "cases": cases,
    }
    written = _write_manifest(output_path, payload)
    lint = lint_manifest_suite(written, strict_materialization=True)
    if not lint["valid"]:
        raise ValueError("; ".join(str(item) for item in lint["errors"]))
    return written
