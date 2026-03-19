#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq  # noqa: E402


def _first_value(row: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
        alt = lowered.get(str(key).lower())
        if alt not in (None, ""):
            return alt
    return default


def _to_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return str(path)


def _table_rows(path: str) -> List[Dict[str, Any]]:
    table = pq.read_table(path)
    return table.to_pylist()


def normalize_swebench_record(row: Dict[str, Any]) -> Dict[str, Any]:
    repo = _to_text(_first_value(row, "repo"))
    return {
        "instance_id": _to_text(_first_value(row, "instance_id", "task_id", "id")),
        "problem_statement": _to_text(_first_value(row, "problem_statement", "prompt", "statement")),
        "repo": repo,
        "repo_clone_url": f"https://github.com/{repo}.git" if repo else "",
        "base_commit": _to_text(_first_value(row, "base_commit")),
        "environment_setup_commit": _to_text(_first_value(row, "environment_setup_commit")),
        "patch": _to_text(_first_value(row, "patch")),
        "test_patch": _to_text(_first_value(row, "test_patch")),
        "hints_text": _to_text(_first_value(row, "hints_text")),
        "created_at": _to_text(_first_value(row, "created_at")),
        "version": _to_text(_first_value(row, "version")),
        "FAIL_TO_PASS": _to_text(_first_value(row, "FAIL_TO_PASS")),
        "PASS_TO_PASS": _to_text(_first_value(row, "PASS_TO_PASS")),
        "difficulty": _to_text(_first_value(row, "difficulty")),
    }


def normalize_gaia_record(row: Dict[str, Any], *, file_name: str = "") -> Dict[str, Any]:
    task_id = _to_text(
        _first_value(
            row,
            "task_id",
            "question_id",
            "Question_ID",
            "id",
            "ID",
            default=file_name.rsplit(".", 1)[0] if file_name else "",
        )
    )
    question = _to_text(_first_value(row, "question", "Question", "prompt", "problem_statement"))
    answer = _to_text(_first_value(row, "final_answer", "Final answer", "answer", "Answer"))
    level = _to_text(_first_value(row, "level", "Level"))
    record: Dict[str, Any] = {
        "question_id": task_id,
        "question": question,
        "final_answer": answer,
        "level": level,
    }
    file_value = _to_text(_first_value(row, "file_name", "file", "attachment", default=file_name))
    if file_value:
        record["file_name"] = file_value
        record["oracle_evidence_file"] = file_value
    return record


def download_swebench_verified(
    *,
    output_records: Path,
    limit: int = 0,
) -> Dict[str, Any]:
    parquet_path = hf_hub_download(
        repo_id="princeton-nlp/SWE-bench_Verified",
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    rows = _table_rows(parquet_path)
    if limit > 0:
        rows = rows[:limit]
    normalized = [normalize_swebench_record(row) for row in rows]
    written = _write_jsonl(output_records, normalized)
    return {
        "corpus": "swebench",
        "record_count": len(normalized),
        "records_path": written,
    }


def download_gaia_validation(
    *,
    output_records: Path,
    attachments_root: Path,
    year: str = "2023",
    split: str = "validation",
    token: str = "",
    limit: int = 0,
) -> Dict[str, Any]:
    token_value = token.strip() or os.environ.get("HF_TOKEN", "").strip() or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
    metadata_filename = f"{year}/{split}/metadata.parquet"
    try:
        metadata_path = hf_hub_download(
            repo_id="gaia-benchmark/GAIA",
            filename=metadata_filename,
            repo_type="dataset",
            token=(token_value or None),
        )
    except GatedRepoError as exc:
        status = {
            "corpus": "gaia",
            "downloaded": False,
            "gated": True,
            "message": str(exc),
            "records_path": str(output_records),
            "attachments_root": str(attachments_root),
        }
        output_records.parent.mkdir(parents=True, exist_ok=True)
        (output_records.parent / "download_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status

    rows = _table_rows(metadata_path)
    if limit > 0:
        rows = rows[:limit]
    attachments_root.mkdir(parents=True, exist_ok=True)
    normalized: List[Dict[str, Any]] = []
    downloaded_files = 0
    for row in rows:
        raw_file_name = _to_text(_first_value(row, "file_name", "file", "attachment"))
        record = normalize_gaia_record(row, file_name=raw_file_name)
        task_id = record["question_id"] or f"gaia_{len(normalized)+1}"
        task_dir = attachments_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        if raw_file_name:
            source_filename = f"{year}/{split}/{raw_file_name}"
            downloaded = hf_hub_download(
                repo_id="gaia-benchmark/GAIA",
                filename=source_filename,
                repo_type="dataset",
                token=(token_value or None),
            )
            target = task_dir / Path(raw_file_name).name
            Path(downloaded).replace(target) if False else target.write_bytes(Path(downloaded).read_bytes())
            record["fixture_dir"] = str(task_dir.resolve())
            downloaded_files += 1
        else:
            record["fixture_dir"] = str(task_dir.resolve())
        normalized.append(record)
    written = _write_jsonl(output_records, normalized)
    return {
        "corpus": "gaia",
        "downloaded": True,
        "gated": False,
        "record_count": len(normalized),
        "attachment_count": downloaded_files,
        "records_path": written,
        "attachments_root": str(attachments_root),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Download public official benchmark corpora into data/official_corpus.")
    ap.add_argument("--corpus", default="all", choices=["all", "swebench", "gaia"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gaia-year", default="2023")
    ap.add_argument("--gaia-split", default="validation")
    ap.add_argument("--gaia-token", default="")
    args = ap.parse_args()

    output: Dict[str, Any] = {"runs": []}
    if args.corpus in {"all", "swebench"}:
        output["runs"].append(
            download_swebench_verified(
                output_records=ROOT / "data" / "official_corpus" / "swebench" / "records.jsonl",
                limit=max(0, int(args.limit)),
            )
        )
    if args.corpus in {"all", "gaia"}:
        output["runs"].append(
            download_gaia_validation(
                output_records=ROOT / "data" / "official_corpus" / "gaia" / "records.jsonl",
                attachments_root=ROOT / "data" / "official_corpus" / "gaia" / "attachments",
                year=str(args.gaia_year),
                split=str(args.gaia_split),
                token=str(args.gaia_token),
                limit=max(0, int(args.limit)),
            )
        )
    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
