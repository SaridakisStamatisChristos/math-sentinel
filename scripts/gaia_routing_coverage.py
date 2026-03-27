from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domains.gaia_ops.backend import (
    _GENERALIZED_RESEARCH_MODES,
    _canonicalize_research_plan,
    _extract_special_research_plan,
)


GENERALIZED_MATRIX_FAMILIES = set(_GENERALIZED_RESEARCH_MODES) | {"arxiv_cross_reference"}


def _case_files(case: Dict[str, Any]) -> List[str]:
    fixture_dir = Path(str(case.get("meta", {}).get("fixture_dir", "") or "")).expanduser()
    if not fixture_dir.is_dir():
        return []
    return sorted(path.name for path in fixture_dir.iterdir() if path.is_file())


def _classify_case(case: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(case.get("prompt", "") or "")
    evidence_files = _case_files(case)
    raw_plan = dict(_extract_special_research_plan(prompt, evidence_files))
    raw_mode = str(raw_plan.get("research_mode", "") or "").strip()
    raw_submode = str(raw_plan.get("solver_submode", "") or "").strip()
    canonical_mode, canonical_submode = _canonicalize_research_plan(raw_mode, raw_submode)
    if not canonical_mode:
        route_class = "fallthrough_default"
    elif canonical_mode in GENERALIZED_MATRIX_FAMILIES:
        route_class = "generalized_family"
    else:
        route_class = "specialized_family"
    return {
        "task_id": str(case.get("task_id", "") or ""),
        "level": str(case.get("meta", {}).get("level", "") or ""),
        "domain": str(case.get("domain", "") or ""),
        "prompt": prompt,
        "evidence_files": evidence_files,
        "raw_mode": raw_mode,
        "raw_submode": raw_submode,
        "canonical_mode": canonical_mode,
        "canonical_submode": canonical_submode,
        "route_class": route_class,
    }


def _short_prompt(prompt: str, *, limit: int = 140) -> str:
    collapsed = " ".join(str(prompt).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def _render_summary(records: List[Dict[str, Any]]) -> List[str]:
    total = len(records)
    generalized = sum(1 for item in records if item["route_class"] == "generalized_family")
    specialized = sum(1 for item in records if item["route_class"] == "specialized_family")
    fallthrough = sum(1 for item in records if item["route_class"] == "fallthrough_default")
    generalized_pct = (100.0 * generalized / total) if total else 0.0
    return [
        f"- Total official GAIA cases analyzed: {total}",
        f"- Explicit generalized family routing: {generalized} ({generalized_pct:.1f}%)",
        f"- Explicit specialized routing outside generalized family set: {specialized}",
        f"- Fallthrough/default routing with no explicit family: {fallthrough}",
    ]


def _render_matrix(records: List[Dict[str, Any]]) -> List[str]:
    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["route_class"],
            record["canonical_mode"] or "(none)",
            record["canonical_submode"] or "-",
        )
        grouped[key].append(record)
    lines = [
        "| Route class | Canonical family | Solver submode | Cases | With files | Example task IDs |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for (route_class, canonical_mode, canonical_submode), items in sorted(
        grouped.items(),
        key=lambda entry: (-len(entry[1]), entry[0][1], entry[0][2]),
    ):
        with_files = sum(1 for item in items if item["evidence_files"])
        sample_ids = ", ".join(item["task_id"] for item in items[:3])
        lines.append(
            f"| {route_class} | {canonical_mode} | {canonical_submode} | {len(items)} | {with_files} | {sample_ids} |"
        )
    return lines


def _render_aliases(records: List[Dict[str, Any]]) -> List[str]:
    aliases: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["raw_mode"] and (
            record["raw_mode"] != record["canonical_mode"]
            or (record["raw_submode"] or "") != (record["canonical_submode"] or "")
        ):
            aliases[
                (
                    record["raw_mode"],
                    record["canonical_mode"] or "(none)",
                    record["canonical_submode"] or "-",
                )
            ].append(record)
    if not aliases:
        return ["No legacy top-level aliases were observed in the manifest classification pass."]
    lines = [
        "| Raw mode | Canonical family | Canonical submode | Cases | Example task IDs |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for (raw_mode, canonical_mode, canonical_submode), items in sorted(
        aliases.items(),
        key=lambda entry: (-len(entry[1]), entry[0][0]),
    ):
        sample_ids = ", ".join(item["task_id"] for item in items[:3])
        lines.append(f"| {raw_mode} | {canonical_mode} | {canonical_submode} | {len(items)} | {sample_ids} |")
    return lines


def _render_fallthrough(records: List[Dict[str, Any]], *, limit: int) -> List[str]:
    fallthrough = [item for item in records if item["route_class"] == "fallthrough_default"]
    if not fallthrough:
        return ["No manifest prompts currently fall through to the default no-family routing path."]
    lines = [
        "| Task ID | Level | Files | Prompt |",
        "| --- | --- | --- | --- |",
    ]
    for item in fallthrough[:limit]:
        files = ", ".join(item["evidence_files"][:4]) or "-"
        lines.append(f"| {item['task_id']} | {item['level'] or '-'} | {files} | {_short_prompt(item['prompt'])} |")
    return lines


def _render_family_examples(records: List[Dict[str, Any]], *, limit: int) -> List[str]:
    lines = [
        "| Canonical family | Solver submode | Example task ID | Prompt |",
        "| --- | --- | --- | --- |",
    ]
    seen: set[tuple[str, str]] = set()
    for item in sorted(records, key=lambda row: (row["canonical_mode"], row["canonical_submode"], row["task_id"])):
        key = (item["canonical_mode"] or "(none)", item["canonical_submode"] or "-")
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"| {key[0]} | {key[1]} | {item['task_id']} | {_short_prompt(item['prompt'])} |"
        )
        if len(seen) >= limit:
            break
    return lines


def build_report(manifest_path: Path, *, fallthrough_limit: int, example_limit: int) -> str:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = list(payload.get("cases", []))
    records = [_classify_case(case) for case in cases]
    sections: List[str] = [
        "# GAIA Routing Coverage Matrix",
        "",
        "## Summary",
        *(_render_summary(records)),
        "",
        "## Canonical Route Matrix",
        *(_render_matrix(records)),
        "",
        "## Raw-To-Canonical Alias Usage",
        *(_render_aliases(records)),
        "",
        "## Family Examples",
        *(_render_family_examples(records, limit=example_limit)),
        "",
        "## Fallthrough Prompts",
        *(_render_fallthrough(records, limit=fallthrough_limit)),
        "",
    ]
    return "\n".join(sections)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a GAIA routing coverage matrix from the official manifest.")
    parser.add_argument(
        "--manifest",
        default="benchmarks/manifests/gaia_full_official.json",
        help="Path to the GAIA manifest JSON file.",
    )
    parser.add_argument(
        "--output",
        default="results/gaia_routing_coverage_matrix.md",
        help="Path to the markdown report to write.",
    )
    parser.add_argument(
        "--fallthrough-limit",
        type=int,
        default=25,
        help="Maximum number of fallthrough prompts to include in the report.",
    )
    parser.add_argument(
        "--example-limit",
        type=int,
        default=20,
        help="Maximum number of canonical family examples to include in the report.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    report = build_report(
        manifest_path,
        fallthrough_limit=max(0, int(args.fallthrough_limit)),
        example_limit=max(1, int(args.example_limit)),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())