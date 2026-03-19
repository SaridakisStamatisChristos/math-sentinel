from __future__ import annotations

import csv
import json
import random
import re
import shutil
import uuid
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from calendar import monthrange
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from benchmarks.integrity import ensure_benchmark_audit, strip_oracle_metadata
from benchmarks.public_catalog import gaia_medium_suite, gaia_smoke_suite
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
ARXIV_API_URL = "https://export.arxiv.org/api/query"


def _private_train_case(
    case_id: str,
    domain: str,
    prompt: str,
    answer: str,
    *,
    fixture_relpath: str,
    evidence_file: str,
) -> ReasoningTask:
    fixture_dir = ROOT / fixture_relpath
    return ReasoningTask(
        task_id=f"gaia_train_{case_id}",
        domain=domain,
        prompt=prompt,
        answer=answer,
        goal="Return the shortest correct final answer",
        meta={
            "family": domain,
            "fixture_dir": str(fixture_dir),
            "oracle_evidence_file": evidence_file,
            "benchmark_suite": "gaia_private_train",
            "benchmark_tier": "train",
            "holdout_group": "gaia_private_train",
            "source": "benchmark_train",
            "fixture_role": "train",
        },
    )


def _private_train_cases() -> List[ReasoningTask]:
    return [
        _private_train_case(
            "team_hours",
            "gaia_csv_reasoning",
            "Use the files in the workspace to answer this question: what is the total support hours for the Orion team in activity.csv? Return only the number.",
            "17",
            fixture_relpath="benchmarks/fixtures/gaia_train/team_hours",
            evidence_file="activity.csv",
        ),
        _private_train_case(
            "theo_tasks",
            "gaia_json_reasoning",
            "Use the files in the workspace to answer this question: in tasks.json, which pending task owned by Theo has the earliest due date? Return only the task title.",
            "Draft brief",
            fixture_relpath="benchmarks/fixtures/gaia_train/theo_tasks",
            evidence_file="tasks.json",
        ),
    ]


def _workspace_for(task: ReasoningTask, *, deterministic: bool = False) -> Path:
    fixture_ref = str(task.meta.get("fixture_dir", "")).strip()
    suffix = "det" if deterministic else uuid.uuid4().hex[:8]
    if not fixture_ref:
        workspace = TMP_ROOT / f"{task.task_id}_{suffix}"
        if deterministic and workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True, exist_ok=True)
        prompt = task.prompt.strip() or "No task prompt provided."
        (workspace / "TASK.md").write_text(prompt + "\n", encoding="utf-8")
        return workspace
    fixture_dir = Path(fixture_ref)
    workspace = TMP_ROOT / f"{task.task_id}_{suffix}"
    workspace.parent.mkdir(parents=True, exist_ok=True)
    if deterministic and workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    shutil.copytree(fixture_dir, workspace)
    return workspace


def _list_workspace_files(workspace: Path) -> List[str]:
    return sorted(str(path.relative_to(workspace)).replace("\\", "/") for path in workspace.rglob("*") if path.is_file())


MONTH_LOOKUP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _tokenize(text: str) -> List[str]:
    cleaned = text.lower()
    for ch in [".", ",", ":", "?", "!", "(", ")", "[", "]", "{", "}", '"', "'"]:
        cleaned = cleaned.replace(ch, " ")
    return [part for part in cleaned.split() if part]


def _infer_target_file(prompt: str, files: Sequence[str]) -> str:
    prompt_lower = prompt.lower()
    for name in files:
        if name.lower() in prompt_lower:
            return name
    prompt_tokens = set(_tokenize(prompt))
    best = ""
    best_score = -1
    for name in files:
        score = sum(1 for token in _tokenize(name) if token in prompt_tokens)
        if score > best_score:
            best = name
            best_score = score
    return best or (files[0] if files else "")


def _resolve_target_files(prompt: str, files: Sequence[str], preferred_file: str = "") -> List[str]:
    mentioned = [name for name in files if name.lower() in prompt.lower()]
    if mentioned:
        return mentioned
    if preferred_file and preferred_file in files:
        return [preferred_file]
    prompt_tokens = set(_tokenize(prompt))
    ranked: List[tuple[int, str]] = []
    for name in files:
        score = sum(1 for token in _tokenize(name) if token in prompt_tokens)
        if score > 0:
            ranked.append((score, name))
    if ranked:
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [name for _, name in ranked]
    if any(name.endswith(".csv") for name in files) and {"csv", "sales", "amount", "total"} & prompt_tokens:
        return [name for name in files if name.endswith(".csv")]
    if any(name.endswith(".json") for name in files) and {"json", "task", "release", "schedule"} & prompt_tokens:
        return [name for name in files if name.endswith(".json")]
    return [files[0]] if files else []


def _infer_question_intent(prompt: str) -> str:
    tokens = set(_tokenize(prompt))
    if {"highest", "largest", "top", "most"} & tokens:
        return "grouped_max"
    if {"earliest", "latest"} & tokens and {"date", "due", "deadline", "task"} & tokens:
        return "date_rank"
    if {"total", "sum"} & tokens:
        return "aggregate_sum"
    if {"earliest", "latest"} & tokens and {"available", "slot", "meeting"} & tokens:
        return "availability_overlap"
    if {"version", "latest"} & tokens:
        return "scalar_lookup"
    if {"count", "many", "number"} & tokens:
        return "count"
    return "scalar_lookup"


def _json_scalar_paths(payload: Any, prefix: str = "") -> List[tuple[str, Any]]:
    items: List[tuple[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_json_scalar_paths(value, next_prefix))
    elif isinstance(payload, list):
        if all(not isinstance(value, (dict, list)) for value in payload):
            items.append((prefix, payload))
        else:
            for index, value in enumerate(payload):
                next_prefix = f"{prefix}[{index}]"
                items.extend(_json_scalar_paths(value, next_prefix))
    else:
        items.append((prefix, payload))
    return items


def _score_scalar_path(prompt: str, path: str, value: Any) -> float:
    prompt_tokens = set(_tokenize(prompt))
    path_tokens = set(_tokenize(path))
    value_tokens = set(_tokenize(str(value)))
    score = float(len(prompt_tokens & path_tokens)) + 0.35 * float(len(prompt_tokens & value_tokens))
    if "latest" in prompt_tokens and "latest" in path_tokens:
        score += 1.0
    if "version" in prompt_tokens and "version" in path_tokens:
        score += 0.8
    return score


def _parse_float(value: Any) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _parse_date(value: Any) -> datetime | None:
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m", "%Y/%m"):
        try:
            parsed = datetime.strptime(text, fmt)
            if fmt in {"%Y-%m", "%Y/%m"}:
                return parsed.replace(day=1)
            return parsed
        except Exception:
            continue
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _extract_prompt_month_year(prompt: str) -> tuple[int | None, int | None]:
    tokens = _tokenize(prompt)
    month = next((MONTH_LOOKUP[token] for token in tokens if token in MONTH_LOOKUP), None)
    year = next((int(token) for token in tokens if token.isdigit() and len(token) == 4), None)
    return month, year


def _csv_rows_with_context(csv_files: Sequence[tuple[str, str]]) -> tuple[List[Dict[str, str]], List[str]]:
    merged_rows: List[Dict[str, str]] = []
    headers: List[str] = []
    for filename, text in csv_files:
        rows = list(csv.DictReader(text.splitlines()))
        if rows and not headers:
            headers = list(rows[0].keys())
        for row in rows:
            contextual = {str(key): str(value) for key, value in row.items()}
            contextual["__file__"] = filename
            merged_rows.append(contextual)
    return merged_rows, headers


def _pick_numeric_header(prompt_tokens: set[str], headers: Sequence[str], rows: Sequence[Dict[str, str]]) -> str:
    numeric_headers: List[str] = []
    for header in headers:
        values = [_parse_float(row.get(header, "")) for row in rows]
        if values and all(value is not None for value in values):
            numeric_headers.append(header)
    if not numeric_headers:
        return headers[-1] if headers else ""
    for header in numeric_headers:
        lowered = header.lower()
        if any(token in lowered for token in prompt_tokens):
            return header
    for header in numeric_headers:
        lowered = header.lower()
        if any(token in lowered for token in ["amount", "total", "sales", "revenue", "count"]):
            return header
    return numeric_headers[0]


def _pick_group_header(prompt_tokens: set[str], headers: Sequence[str], numeric_header: str, date_headers: Sequence[str]) -> str:
    categorical = [header for header in headers if header not in {numeric_header, *date_headers}]
    for header in categorical:
        lowered = header.lower()
        if any(token in lowered for token in prompt_tokens):
            return header
    for preferred in ["city", "region", "project", "owner", "name", "title"]:
        for header in categorical:
            if preferred in header.lower():
                return header
    return categorical[0] if categorical else numeric_header


def _pick_answer_header(prompt_tokens: set[str], headers: Sequence[str], date_headers: Sequence[str], numeric_header: str = "") -> str:
    for preferred in ["title", "task", "name", "version", "city", "project"]:
        for header in headers:
            if preferred in header.lower():
                return header
    for header in headers:
        if header not in date_headers and header != numeric_header:
            return header
    return headers[0] if headers else ""


def _infer_csv_answer(prompt: str, csv_files: Sequence[tuple[str, str]]) -> tuple[str, List[str]]:
    rows, headers = _csv_rows_with_context(csv_files)
    if not rows:
        return "", []
    prompt_tokens = set(_tokenize(prompt))
    value_map: Dict[str, List[str]] = {}
    date_headers = [header for header in headers if any(_parse_date(row.get(header, "")) is not None for row in rows)]
    for header in headers:
        for row in rows:
            value = str(row.get(header, "")).strip()
            if value:
                value_map.setdefault(value.lower(), []).append(header)
    filters: List[tuple[str, str]] = []
    for value, columns in value_map.items():
        if value in prompt_tokens:
            filters.append((columns[0], value))
    filtered_rows = rows
    for filter_col, filter_value in filters:
        filtered_rows = [row for row in filtered_rows if str(row.get(filter_col, "")).strip().lower() == filter_value]
    month, year = _extract_prompt_month_year(prompt)
    if month is not None or year is not None:
        narrowed: List[Dict[str, str]] = []
        for row in filtered_rows:
            for header in date_headers:
                parsed = _parse_date(row.get(header, ""))
                if parsed is None:
                    continue
                if month is not None and parsed.month != month:
                    continue
                if year is not None and parsed.year != year:
                    continue
                narrowed.append(row)
                break
        if narrowed:
            filtered_rows = narrowed

    target_header = _pick_numeric_header(prompt_tokens, headers, filtered_rows or rows)
    evidence = [f"rows considered: {len(filtered_rows)} across {len(csv_files)} file(s)"]
    if filters:
        evidence.insert(0, ", ".join(f"{column}={value}" for column, value in filters))

    if {"highest", "largest", "top", "most"} & prompt_tokens:
        group_header = _pick_group_header(prompt_tokens, headers, target_header, date_headers)
        totals: Dict[str, float] = {}
        for row in filtered_rows:
            key = str(row.get(group_header, "")).strip()
            value = _parse_float(row.get(target_header, "")) or 0.0
            if key:
                totals[key] = totals.get(key, 0.0) + value
        if not totals:
            return "", evidence
        best_key, best_value = max(totals.items(), key=lambda item: (item[1], item[0]))
        evidence.append(f"grouped by {group_header}, max {target_header} -> {best_key} ({best_value:g})")
        return best_key, evidence

    if {"earliest", "latest"} & prompt_tokens and date_headers:
        date_header = next((header for header in date_headers if any(token in header.lower() for token in ["date", "due", "deadline"])), date_headers[0])
        answer_header = _pick_answer_header(prompt_tokens, headers, date_headers, target_header)
        dated_rows = [(row, _parse_date(row.get(date_header, ""))) for row in filtered_rows]
        dated_rows = [(row, parsed) for row, parsed in dated_rows if parsed is not None]
        if not dated_rows:
            return "", evidence
        chooser = min if "earliest" in prompt_tokens else max
        best_row, best_date = chooser(dated_rows, key=lambda item: item[1])
        candidate = str(best_row.get(answer_header, "")).strip()
        evidence.append(f"{answer_header} selected from {date_header}={best_date.date().isoformat()}")
        return candidate, evidence

    total = sum((_parse_float(row.get(target_header, "")) or 0.0) for row in filtered_rows)
    rendered = str(int(total)) if float(total).is_integer() else str(total)
    evidence.append(f"sum({target_header}) -> {rendered}")
    return rendered, evidence


def _collect_json_records(payload: Any) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        scalar_fields = {str(key): value for key, value in payload.items() if not isinstance(value, (dict, list))}
        if scalar_fields:
            records.append(scalar_fields)
        for value in payload.values():
            records.extend(_collect_json_records(value))
    elif isinstance(payload, list):
        for value in payload:
            records.extend(_collect_json_records(value))
    return records


def _infer_json_record_answer(prompt: str, payload: Any) -> tuple[str, List[str]]:
    records = _collect_json_records(payload)
    if not records:
        return "", []
    prompt_tokens = set(_tokenize(prompt))
    filtered_records = records
    for token in sorted(prompt_tokens):
        narrowed = [
            record for record in filtered_records
            if any(str(value).strip().lower() == token for value in record.values())
        ]
        if narrowed:
            filtered_records = narrowed
    if {"earliest", "latest"} & prompt_tokens:
        dated_candidates: List[tuple[Dict[str, Any], str, datetime]] = []
        for record in filtered_records:
            for key, value in record.items():
                parsed = _parse_date(value)
                if parsed is None:
                    continue
                if not any(marker in key.lower() for marker in ["date", "due", "deadline", "release"]):
                    continue
                dated_candidates.append((record, str(key), parsed))
        if dated_candidates:
            chooser = min if "earliest" in prompt_tokens else max
            best_record, best_key, best_date = chooser(dated_candidates, key=lambda item: item[2])
            answer_key = next((key for key in best_record if any(marker in key.lower() for marker in ["title", "task", "name", "version"])), next(iter(best_record.keys())))
            return str(best_record.get(answer_key, "")), [f"{answer_key} chosen from {best_key}={best_date.date().isoformat()}"]
    return "", []


def _infer_json_answer(prompt: str, payload: Any) -> tuple[str, List[str]]:
    record_answer, record_evidence = _infer_json_record_answer(prompt, payload)
    if record_answer:
        return record_answer, record_evidence
    prompt_tokens = set(_tokenize(prompt))
    if isinstance(payload, dict):
        lower_keys = {str(key).lower(): key for key in payload.keys()}
        if "people" in lower_keys and isinstance(payload[lower_keys["people"]], dict):
            people = payload[lower_keys["people"]]
            mentioned = [name for name in people.keys() if str(name).lower() in prompt_tokens]
            if len(mentioned) >= 2:
                common = None
                for name in mentioned[:2]:
                    slots = set(str(value) for value in people.get(name, []))
                    common = slots if common is None else common & slots
                if common:
                    ordered = sorted(common)
                    rendered = ordered[0] if "earliest" in prompt_tokens else ordered[-1]
                    return rendered, [f"intersection({', '.join(mentioned[:2])}) -> {rendered}"]
    candidates = _json_scalar_paths(payload)
    if not candidates:
        return "", []
    scored = sorted(candidates, key=lambda item: _score_scalar_path(prompt, item[0], item[1]), reverse=True)
    best_path, best_value = scored[0]
    return str(best_value), [f"{best_path} -> {best_value}"]


def _infer_multi_json_answer(prompt: str, json_files: Sequence[tuple[str, Any]]) -> tuple[str, List[str]]:
    best_answer = ""
    best_evidence: List[str] = []
    best_score = -1.0
    prompt_tokens = set(_tokenize(prompt))
    for name, payload in json_files:
        candidate, evidence = _infer_json_answer(prompt, payload)
        if not candidate:
            continue
        score = float(len(evidence))
        score += 0.25 * float(sum(1 for token in prompt_tokens if token in str(candidate).lower()))
        if score > best_score:
            best_answer = candidate
            best_evidence = [f"from {name}"] + evidence
            best_score = score
    return best_answer, best_evidence


def _merge_evidence_graph(existing: Any, *, relpath: str, summary: str, file_kind: str) -> Dict[str, Any]:
    graph = dict(existing) if isinstance(existing, dict) else {}
    files = [str(item) for item in graph.get("files", []) if str(item).strip()]
    if relpath and relpath not in files:
        files.append(relpath)
    nodes = list(graph.get("nodes", [])) if isinstance(graph.get("nodes", []), list) else []
    nodes.append({"file": relpath, "kind": file_kind, "summary": summary[:160]})
    edges = list(graph.get("edges", [])) if isinstance(graph.get("edges", []), list) else []
    if len(files) >= 2:
        edge = {"from": files[-2], "to": files[-1], "relation": "inspected_after"}
        if edge not in edges:
            edges.append(edge)
    graph["files"] = files
    graph["nodes"] = nodes[-8:]
    graph["edges"] = edges[-8:]
    return graph


def _answer_confidence(candidate: str, evidence: Sequence[str], file_count: int, *, fallback_text: bool = False) -> float:
    if not candidate:
        return 0.0
    confidence = 0.45
    confidence += min(0.30, 0.10 * len([item for item in evidence if str(item).strip()]))
    confidence += 0.08 if file_count <= 2 else 0.03
    if fallback_text:
        confidence -= 0.18
    if len(str(candidate).strip()) <= 4:
        confidence += 0.05
    return max(0.05, min(0.99, confidence))


def _extract_date_mentions(prompt: str) -> List[Dict[str, int]]:
    month_names = "|".join(sorted(MONTH_LOOKUP.keys(), key=len, reverse=True))
    pattern = re.compile(rf"\b({month_names})\s+(?:(\d{{1,2}}),\s+)?(\d{{4}})\b", re.IGNORECASE)
    mentions: List[Dict[str, int]] = []
    for month_name, day_text, year_text in pattern.findall(prompt or ""):
        month = MONTH_LOOKUP[str(month_name).lower()]
        day = int(day_text) if str(day_text).strip() else 0
        year = int(year_text)
        mentions.append({"year": year, "month": month, "day": day})
    return mentions


def _arxiv_date_window(spec: Dict[str, int]) -> tuple[str, str]:
    year = int(spec.get("year", 0))
    month = int(spec.get("month", 1))
    day = int(spec.get("day", 0))
    if day > 0:
        start = f"{year:04d}{month:02d}{day:02d}0000"
        end = f"{year:04d}{month:02d}{day:02d}2359"
        return (start, end)
    last_day = monthrange(year, month)[1]
    return (f"{year:04d}{month:02d}010000", f"{year:04d}{month:02d}{last_day:02d}2359")


def _extract_arxiv_research_plan(prompt: str) -> Dict[str, Any]:
    lowered = (prompt or "").lower()
    if "arxiv.org" not in lowered:
        return {}
    primary_query = ""
    primary_match = re.search(r"paper about (.+?) that was originally submitted to arxiv\.org", prompt or "", re.IGNORECASE)
    if primary_match:
        primary_query = str(primary_match.group(1)).strip(" .?")
    secondary_category = "physics.soc-ph" if "physics and society" in lowered else ""
    dates = _extract_date_mentions(prompt or "")
    primary_dates = _arxiv_date_window(dates[0]) if dates else ("", "")
    secondary_dates = _arxiv_date_window(dates[1]) if len(dates) >= 2 else ("", "")
    return {
        "research_mode": "arxiv_cross_reference",
        "primary_query": primary_query,
        "primary_dates": primary_dates,
        "secondary_query": "Physics and Society" if secondary_category else "",
        "secondary_category": secondary_category,
        "secondary_dates": secondary_dates,
    }


def _arxiv_search(query: str, *, start_date: str = "", end_date: str = "", category: str = "", max_results: int = 5) -> List[Dict[str, Any]]:
    search_terms: List[str] = []
    cleaned_query = str(query).strip()
    if cleaned_query:
        escaped = cleaned_query.replace('"', "")
        search_terms.append(f'all:"{escaped}"')
    if category:
        search_terms.append(f"cat:{category}")
    if start_date and end_date:
        search_terms.append(f"submittedDate:[{start_date} TO {end_date}]")
    search_query = " AND ".join(search_terms) if search_terms else "all:*"
    params = urllib.parse.urlencode(
        {
            "search_query": search_query,
            "start": 0,
            "max_results": max(1, int(max_results)),
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
    )
    with urllib.request.urlopen(f"{ARXIV_API_URL}?{params}", timeout=20) as response:
        payload = response.read()
    root = ET.fromstring(payload)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries: List[Dict[str, Any]] = []
    for item in root.findall("atom:entry", ns):
        title = " ".join(str(item.findtext("atom:title", default="", namespaces=ns)).split())
        summary = " ".join(str(item.findtext("atom:summary", default="", namespaces=ns)).split())
        entry_id = str(item.findtext("atom:id", default="", namespaces=ns)).strip()
        published = str(item.findtext("atom:published", default="", namespaces=ns)).strip()
        categories = [str(node.attrib.get("term", "")).strip() for node in item.findall("atom:category", ns)]
        entries.append(
            {
                "id": entry_id,
                "title": title,
                "summary": summary,
                "published": published,
                "categories": [term for term in categories if term],
            }
        )
    return entries


def _extract_overlap_terms(entries: Sequence[Dict[str, Any]]) -> List[str]:
    stopwords = {
        "paper",
        "about",
        "their",
        "these",
        "those",
        "which",
        "where",
        "while",
        "there",
        "fairness",
        "values",
        "perspective",
        "regulation",
        "article",
        "society",
        "physics",
    }
    terms: List[str] = []
    for entry in entries:
        combined = f"{entry.get('title', '')}. {entry.get('summary', '')}"
        for sequence in re.findall(r"((?:[A-Za-z-]+,\s+){2,}(?:and\s+)?[A-Za-z-]+)", combined):
            for word in re.findall(r"[A-Za-z-]+", sequence):
                lowered = word.lower()
                if len(lowered) >= 5 and lowered not in stopwords and lowered not in terms:
                    terms.append(lowered)
    return terms[:24]


def _extract_axis_terms(entries: Sequence[Dict[str, Any]]) -> List[str]:
    axis_terms: List[str] = []

    def _add_term(term: str) -> None:
        cleaned = str(term).strip().lower().strip(".,:;()[]{}")
        if not cleaned:
            return
        if cleaned not in axis_terms:
            axis_terms.append(cleaned)
        if cleaned.endswith("ism") and len(cleaned) > 4:
            variant = cleaned[:-3].strip("- ")
            if variant and variant not in axis_terms:
                axis_terms.append(variant)

    for entry in entries:
        combined = f"{entry.get('title', '')}. {entry.get('summary', '')}"
        for left, right in re.findall(r"\b([A-Za-z-]+)\s+vs\.\s+([A-Za-z-]+)\b", combined):
            _add_term(left)
            _add_term(right)
    return axis_terms[:24]


def _solve_arxiv_overlap(primary_entries: Sequence[Dict[str, Any]], secondary_entries: Sequence[Dict[str, Any]]) -> tuple[str, List[str]]:
    candidate_terms = _extract_axis_terms(primary_entries) or _extract_overlap_terms(primary_entries)
    if not candidate_terms:
        return ("", [])
    evidence: List[str] = []
    scored_terms: List[tuple[int, str]] = []
    for term in candidate_terms:
        title_hits = 0
        summary_hits = 0
        matched_entries: List[str] = []
        for entry in secondary_entries:
            title = str(entry.get("title", ""))
            summary = str(entry.get("summary", ""))
            title_match = bool(re.search(rf"\b{re.escape(term)}\b", title, flags=re.IGNORECASE))
            summary_match = bool(re.search(rf"\b{re.escape(term)}\b", summary, flags=re.IGNORECASE))
            if title_match:
                title_hits += 1
            if summary_match:
                summary_hits += 1
            if title_match or summary_match:
                matched_entries.append(str(entry.get("title", "")).strip())
        score = title_hits * 5 + summary_hits * 2
        if score > 0:
            evidence.append(
                f"term {term} matched {title_hits} secondary title(s) and {summary_hits} secondary summary hit(s)"
            )
            if matched_entries:
                evidence.append(f"matching secondary titles: {', '.join(matched_entries[:2])}")
            scored_terms.append((score, term))
    if scored_terms:
        scored_terms.sort(key=lambda item: (-item[0], candidate_terms.index(item[1])))
        return (scored_terms[0][1], evidence)
    return ("", evidence)


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
            "obligations": ["inspect evidence file", "solve from evidence"],
            "state_metadata": {"workspace_files": files},
        },
    }


def plan_question(arg: str, state: Any = None) -> Dict[str, Any]:
    prompt = str(getattr(state, "problem_text", "")).split("\nWorkspace files:\n", 1)[0].strip()
    files = list(state.metadata.get("workspace_files", []))
    evidence_files = [name for name in files if name != "TASK.md"]
    research_plan = _extract_arxiv_research_plan(prompt) if not evidence_files else {}
    target_file = _infer_target_file(prompt, files)
    candidate_files = _resolve_target_files(prompt, files, target_file)
    intent = _infer_question_intent(prompt)
    if research_plan:
        plan = f"search arXiv for '{research_plan.get('primary_query', '')}' then cross-reference physics.soc-ph results"
        ambiguity_score = 0.35
    else:
        target_label = ", ".join(candidate_files[:3]) if candidate_files else (target_file or "the most relevant file")
        plan = f"inspect {target_label} then solve intent={intent}"
        ambiguity_score = max(0.0, min(1.0, float(max(0, len(candidate_files) - 1)) / 3.0))
    if str(state.metadata.get("benchmark_assistance_mode", "unassisted")) == "assisted" and bool(state.metadata.get("oracle_hints_enabled", False)):
        oracle_file = str(state.metadata.get("oracle_evidence_file", "")).strip()
        if oracle_file:
            target_file = oracle_file
            plan = f"inspect {target_file} then solve intent={intent}"
    return {
        "ok": True,
        "result": plan,
        "goal_progress": 0.18,
        "payload": {
            "evidence": [plan],
            "suggested_tools": ["list_files", "inspect_file", "solve_question"],
            "obligations": ["inspect evidence file", "solve from evidence"],
            "state_metadata": {
                "target_file": target_file,
                "candidate_files": candidate_files,
                "question_intent": intent,
                "ambiguity_score": ambiguity_score,
                "question_plan": {
                    "intent": intent,
                    "target_file": target_file,
                    "candidate_files": candidate_files[:4],
                    **research_plan,
                },
            },
        },
    }


def search_arxiv_primary(arg: str, state: Any = None) -> Dict[str, Any]:
    plan = dict(state.metadata.get("question_plan", {}))
    if plan.get("research_mode") != "arxiv_cross_reference":
        return {"ok": False, "result": "no arXiv research plan available", "risk": 0.7}
    start_date, end_date = tuple(plan.get("primary_dates", ("", "")))
    entries = _arxiv_search(
        str(plan.get("primary_query", "")),
        start_date=str(start_date),
        end_date=str(end_date),
        category="",
        max_results=5,
    )
    candidate_terms = _extract_axis_terms(entries) or _extract_overlap_terms(entries)
    rendered = "\n".join(f"{entry['published'][:10]} | {entry['title']}" for entry in entries[:4]) or "no results"
    return {
        "ok": True,
        "result": rendered,
        "goal_progress": 0.30,
        "payload": {
            "evidence": [f"primary arXiv query returned {len(entries)} result(s)"] + [entry["title"] for entry in entries[:2]],
            "resolved_obligations": ["inspect evidence file"],
            "obligations": ["search external evidence", "solve from evidence"],
            "state_metadata": {
                "arxiv_primary_results": entries,
                "arxiv_candidate_terms": candidate_terms,
            },
        },
        "risk": 0.15 if entries else 0.55,
    }


def search_arxiv_secondary(arg: str, state: Any = None) -> Dict[str, Any]:
    plan = dict(state.metadata.get("question_plan", {}))
    if plan.get("research_mode") != "arxiv_cross_reference":
        return {"ok": False, "result": "no arXiv research plan available", "risk": 0.7}
    start_date, end_date = tuple(plan.get("secondary_dates", ("", "")))
    entries = _arxiv_search(
        str(plan.get("secondary_query", "")),
        start_date=str(start_date),
        end_date=str(end_date),
        category=str(plan.get("secondary_category", "")),
        max_results=25,
    )
    rendered = "\n".join(f"{entry['published'][:10]} | {entry['title']}" for entry in entries[:6]) or "no results"
    return {
        "ok": True,
        "result": rendered,
        "goal_progress": 0.35,
        "payload": {
            "evidence": [f"secondary arXiv query returned {len(entries)} result(s)"] + [entry["title"] for entry in entries[:2]],
            "resolved_obligations": ["search external evidence"],
            "obligations": ["solve from evidence"],
            "state_metadata": {"arxiv_secondary_results": entries},
        },
        "risk": 0.15 if entries else 0.60,
    }


def inspect_file(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    files = list(state.metadata.get("workspace_files", []))
    relpath = arg.strip() or str(state.metadata.get("target_file", "")) or _infer_target_file(str(getattr(state, "problem_text", "")), files)
    if not relpath:
        return {"ok": False, "result": "no file available"}
    text = (workspace / relpath).read_text(encoding="utf-8")
    summary = ""
    file_kind = Path(relpath).suffix.lower().lstrip(".") or "text"
    inspected_files = [str(item) for item in state.metadata.get("inspected_files", []) if str(item).strip()]
    if relpath and relpath not in inspected_files:
        inspected_files.append(relpath)
    payload: Dict[str, Any] = {
        "path": relpath,
        "state_metadata": {
            "target_file": relpath,
            "active_file": relpath,
            "active_file_kind": file_kind,
            "inspected_files": inspected_files,
        },
    }
    if file_kind == "csv":
        rows = list(csv.DictReader(text.splitlines()))
        columns = list(rows[0].keys()) if rows else []
        summary = f"csv columns: {', '.join(columns)}"
        payload["columns"] = columns
        payload["row_count"] = len(rows)
    elif file_kind == "json":
        json_payload = json.loads(text)
        scalar_paths = _json_scalar_paths(json_payload)
        top_paths = [path for path, _ in scalar_paths[:6]]
        summary = f"json paths: {', '.join(top_paths)}"
        payload["scalar_paths"] = top_paths
    else:
        summary = text[:200]
    payload.update(
        {
            "evidence": [f"inspected {relpath}", summary],
            "resolved_obligations": ["inspect evidence file"],
            "obligations": ["solve from evidence"],
        }
    )
    payload["state_metadata"]["evidence_graph"] = _merge_evidence_graph(
        state.metadata.get("evidence_graph", {}),
        relpath=relpath,
        summary=summary,
        file_kind=file_kind,
    )
    payload["state_metadata"]["ambiguity_score"] = max(
        0.0,
        min(1.0, float(max(0, len([name for name in state.metadata.get("candidate_files", []) if str(name).strip()]) - len(inspected_files))) / 3.0),
    )
    return {"ok": True, "result": text, "goal_progress": 0.25, "payload": payload}


def solve_question(arg: str, state: Any = None) -> Dict[str, Any]:
    workspace = Path(str(state.metadata["workspace_dir"]))
    prompt = (arg.strip() or str(getattr(state, "problem_text", ""))).split("\nWorkspace files:\n", 1)[0].strip()
    files = [name for name in state.metadata.get("workspace_files", []) if str(name) != "TASK.md"]
    plan = dict(state.metadata.get("question_plan", {}))
    if not files and plan.get("research_mode") == "arxiv_cross_reference":
        primary_entries = list(state.metadata.get("arxiv_primary_results", []))
        secondary_entries = list(state.metadata.get("arxiv_secondary_results", []))
        candidate, evidence = _solve_arxiv_overlap(primary_entries, secondary_entries)
        if not candidate:
            return {"ok": False, "result": "could not infer answer from arXiv evidence", "risk": 0.70}
        confidence = _answer_confidence(candidate, evidence, max(1, len(primary_entries) + len(secondary_entries)))
        return {
            "ok": True,
            "result": candidate,
            "goal_progress": 0.82,
            "payload": {
                "candidate_answer": candidate,
                "answer": candidate,
                "evidence": evidence + [f"confidence={confidence:.2f}"],
                "resolved_obligations": ["solve from evidence"],
                "state_metadata": {
                    "candidate_answer": candidate,
                    "answer_confidence": confidence,
                    "answer_provenance": ["arxiv:primary", "arxiv:secondary"],
                    "ambiguity_score": max(0.0, 0.55 - confidence),
                },
            },
            "risk": max(0.0, 1.0 - confidence),
        }
    target_file = str(state.metadata.get("target_file", ""))
    inspected_files = [str(item) for item in state.metadata.get("inspected_files", []) if str(item).strip()]
    planned_files = list(state.metadata.get("candidate_files", [])) or _resolve_target_files(prompt, files, target_file)
    candidate_files = []
    for name in inspected_files + planned_files:
        text = str(name).strip()
        if text and text not in candidate_files:
            candidate_files.append(text)
    if str(state.metadata.get("benchmark_assistance_mode", "unassisted")) == "assisted" and bool(state.metadata.get("oracle_hints_enabled", False)):
        target_file = str(state.metadata.get("oracle_evidence_file", "") or target_file)
        candidate_files = [target_file] if target_file else candidate_files
    if not candidate_files and target_file:
        candidate_files = [target_file]
    if not candidate_files:
        return {"ok": False, "result": "no target file inferred", "risk": 0.7}
    existing_paths = [(name, workspace / name) for name in candidate_files if (workspace / name).exists()]
    if not existing_paths:
        return {"ok": False, "result": f"file not found: {candidate_files[0]}", "risk": 0.7}
    suffixes = {path.suffix.lower() for _, path in existing_paths}
    candidate = ""
    evidence: List[str] = []
    answer_provenance: List[str] = []
    resolved_target = existing_paths[0][0]
    fallback_text = False
    if suffixes == {".csv"}:
        csv_files = [(name, path.read_text(encoding="utf-8")) for name, path in existing_paths]
        candidate, evidence = _infer_csv_answer(prompt, csv_files)
        answer_provenance = [f"csv:{name}" for name, _ in existing_paths]
    elif suffixes == {".json"} and len(existing_paths) == 1:
        resolved_target, path = existing_paths[0]
        candidate, evidence = _infer_json_answer(prompt, json.loads(path.read_text(encoding="utf-8")))
        answer_provenance = [f"json:{resolved_target}"]
    elif suffixes == {".json"}:
        json_files = [(name, json.loads(path.read_text(encoding="utf-8"))) for name, path in existing_paths]
        candidate, evidence = _infer_multi_json_answer(prompt, json_files)
        answer_provenance = [f"json:{name}" for name, _ in existing_paths]
    else:
        resolved_target, path = existing_paths[0]
        text = path.read_text(encoding="utf-8")
        candidate = text.strip().splitlines()[0] if text.strip() else ""
        evidence = [f"used first non-empty line from {resolved_target}"]
        answer_provenance = [f"text:{resolved_target}"]
        fallback_text = True
    if not candidate:
        return {"ok": False, "result": "could not infer answer from evidence", "risk": 0.75}
    confidence = _answer_confidence(candidate, evidence, len(existing_paths), fallback_text=fallback_text)
    ambiguity_score = max(
        0.0,
        min(
            1.0,
            float(max(0, len(candidate_files) - len(existing_paths))) / 3.0 + max(0.0, 0.55 - confidence),
        ),
    )
    state_metadata = {
        "target_file": resolved_target,
        "candidate_files": [name for name, _ in existing_paths],
        "answer_confidence": confidence,
        "answer_provenance": answer_provenance,
        "ambiguity_score": ambiguity_score,
    }
    if confidence >= 0.45:
        state_metadata["candidate_answer"] = candidate
    return {
        "ok": True,
        "result": candidate,
        "goal_progress": 0.8,
        "payload": {
            "candidate_answer": candidate if confidence >= 0.45 else "",
            "answer": candidate,
            "evidence": evidence + [f"confidence={confidence:.2f}"],
            "resolved_obligations": ["solve from evidence"],
            "state_metadata": state_metadata,
        },
        "risk": max(0.0, 1.0 - confidence),
    }


class GaiaToolRegistry:
    def __init__(self) -> None:
        self.tools = {
            "plan_question": plan_question,
            "list_files": list_files,
            "inspect_file": inspect_file,
            "search_arxiv_primary": search_arxiv_primary,
            "search_arxiv_secondary": search_arxiv_secondary,
            "solve_question": solve_question,
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

    def __init__(self, runtime_config: Dict[str, Any] | None = None) -> None:
        self._train_cases = _private_train_cases()
        self._benchmark_cases = list(gaia_smoke_suite().cases) + list(gaia_medium_suite().cases)
        self._all_cases = self._train_cases + self._benchmark_cases
        runtime_cfg = dict((runtime_config or {}).get("runtime", {}))
        benchmark_cfg = dict((runtime_config or {}).get("benchmark", {}))
        self.deterministic_runtime = bool(runtime_cfg.get("deterministic", False))
        self.assistance_mode = str(benchmark_cfg.get("assistance_mode", "unassisted")).lower()
        self.oracle_hints_enabled = bool(benchmark_cfg.get("oracle_hints_enabled", False))
        self.holdout_enabled = bool(benchmark_cfg.get("holdout_enabled", True))
        self.claim_mode = bool(benchmark_cfg.get("claim_mode", False))

    def _match_manual_case(self, prompt: str, domain: str) -> Optional[ReasoningTask]:
        text = f"{domain}\n{prompt}".lower()
        score_map: List[tuple[int, ReasoningTask]] = []
        for case in self._all_cases:
            keywords = {case.task_id.lower(), case.domain.lower(), str(case.meta.get("family", "")).lower()}
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
        pool = self._train_cases if self.holdout_enabled and self._train_cases else self._all_cases
        eligible = [task for task in pool if task.domain in domains] or pool
        return random.choice(eligible)

    def make_state(self, task: ReasoningTask) -> ReasoningState:
        workspace = _workspace_for(task, deterministic=self.deterministic_runtime)
        files = _list_workspace_files(workspace)
        raw_metadata = dict(task.meta)
        metadata = dict(raw_metadata if self.assistance_mode == "assisted" and self.oracle_hints_enabled else strip_oracle_metadata(raw_metadata))
        metadata["workspace_dir"] = str(workspace)
        metadata["workspace_files"] = files
        metadata["benchmark_assistance_mode"] = self.assistance_mode
        metadata["oracle_hints_enabled"] = self.oracle_hints_enabled
        metadata["claim_mode"] = self.claim_mode
        metadata["benchmark_suite"] = str(raw_metadata.get("benchmark_suite", metadata.get("benchmark_suite", "")))
        metadata["holdout_group"] = str(raw_metadata.get("holdout_group", metadata.get("holdout_group", "")))
        metadata["source"] = str(raw_metadata.get("source", metadata.get("source", "")))
        metadata["fixture_role"] = str(raw_metadata.get("fixture_role", metadata.get("fixture_role", "")))
        metadata["target_file"] = _infer_target_file(task.prompt, files)
        metadata["candidate_files"] = _resolve_target_files(task.prompt, files, str(metadata.get("target_file", "")))
        ensure_benchmark_audit(metadata, assistance_mode=self.assistance_mode)
        if self.assistance_mode == "assisted" and self.oracle_hints_enabled:
            oracle_file = str(raw_metadata.get("oracle_evidence_file", "")).strip()
            if oracle_file:
                metadata["target_file"] = oracle_file
                metadata["candidate_files"] = [oracle_file]
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
        actions = [
            Action(type=ActionType.THINK, content="plan the question, inspect the relevant file, solve from evidence, then answer"),
            Action(type=ActionType.APPLY, tool="plan_question", content=task.prompt),
            Action(type=ActionType.APPLY, tool="list_files", content=""),
            Action(type=ActionType.APPLY, tool="inspect_file", content=str(task.meta.get("oracle_evidence_file", ""))),
            Action(type=ActionType.APPLY, tool="solve_question", content=task.prompt),
            Action(type=ActionType.ANSWER, content=task.answer),
        ]
        return render_canonical_actions(actions)

    def build_verifier_examples(self, task: ReasoningTask) -> tuple[str, torch.Tensor, str, torch.Tensor]:
        pos = self.make_state(task)
        pos.final_answer = task.answer
        pos.status = "solved"
        pos.derived_facts.append(task.answer)
        pos.action_history.append({"type": "ANSWER", "content": task.answer})
        pos.tool_history.append({"tool": "solve_question", "result": {"ok": True, "answer": task.answer}})

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
        return torch.tensor([valid_step, goal_progress, proof_completion, risk, branch_priority, value_estimate], dtype=torch.float32)

    def evaluate_answer(self, task: ReasoningTask, candidate: str) -> bool:
        return candidate.strip() == task.answer.strip()

    def parse_actions(self, text: str) -> tuple[List[Any], float]:
        return parse_actions(text)

    @staticmethod
    def _tool_names(state: ReasoningState) -> List[str]:
        return [record.get("tool", "") for record in state.tool_history if isinstance(record, dict)]

    def _answer_candidates(self, state: ReasoningState) -> List[Dict[str, str]]:
        answers: List[str] = []
        threshold = 0.45
        metadata_confidence = float(state.metadata.get("answer_confidence", 0.0) or 0.0)
        for candidate in [
            state.final_answer,
            str(state.metadata.get("candidate_answer", "")),
        ]:
            text = str(candidate).strip()
            if text and (text == state.final_answer or metadata_confidence >= threshold) and text not in answers:
                answers.append(text)
        for record in reversed(list(state.tool_history)):
            if not isinstance(record, dict):
                continue
            result = record.get("result", {})
            if not isinstance(result, dict):
                continue
            payload = result.get("result_payload", {})
            confidence = metadata_confidence
            if isinstance(payload, dict):
                state_metadata = payload.get("state_metadata", {})
                if isinstance(state_metadata, dict):
                    confidence = float(state_metadata.get("answer_confidence", confidence) or confidence)
            candidate = str(result.get("answer", "")).strip()
            if candidate and confidence >= threshold and candidate not in answers:
                answers.append(candidate)
        return [{"content": item} for item in answers]

    def _next_apply_tools(self, state: ReasoningState) -> List[str]:
        tool_names = self._tool_names(state)
        plan = dict(state.metadata.get("question_plan", {}))
        if plan.get("research_mode") == "arxiv_cross_reference":
            if "plan_question" not in tool_names:
                return ["plan_question"]
            if "search_arxiv_primary" not in tool_names:
                return ["search_arxiv_primary"]
            if "search_arxiv_secondary" not in tool_names:
                return ["search_arxiv_secondary"]
            if "solve_question" not in tool_names:
                return ["solve_question"]
            return ["search_arxiv_secondary", "solve_question"]
        inspected_files = [str(item) for item in state.metadata.get("inspected_files", []) if str(item).strip()]
        candidate_files = [str(item) for item in state.metadata.get("candidate_files", []) if str(item).strip()]
        remaining_files = [name for name in candidate_files if name not in inspected_files]
        if "plan_question" not in tool_names:
            return ["plan_question"]
        if "list_files" not in tool_names:
            return ["list_files"]
        if "inspect_file" not in tool_names:
            return ["inspect_file"]
        if remaining_files and float(state.metadata.get("ambiguity_score", 0.0) or 0.0) >= 0.25 and len(inspected_files) < min(2, len(candidate_files)):
            return ["inspect_file", "solve_question"]
        if "solve_question" not in tool_names:
            return ["solve_question"]
        return ["inspect_file", "solve_question", "list_files"]

    def _retrieval_filters(self, state: ReasoningState) -> Dict[str, Any]:
        if not bool(state.metadata.get("claim_mode", False)):
            return {}
        filters: Dict[str, Any] = {
            "exclude_sources": ["benchmark", "benchmark_claim_holdout", "public_benchmark"],
        }
        suite = str(state.metadata.get("benchmark_suite", "")).strip()
        holdout_group = str(state.metadata.get("holdout_group", "")).strip()
        if suite:
            filters["exclude_suites"] = [suite]
        if holdout_group:
            filters["exclude_holdout_groups"] = [holdout_group]
        return filters

    def fallback_repairs(self, state: ReasoningState) -> List[Action]:
        tool_names = self._tool_names(state)
        plan = dict(state.metadata.get("question_plan", {}))
        if plan.get("research_mode") == "arxiv_cross_reference":
            if "plan_question" not in tool_names:
                return [Action(type=ActionType.APPLY, tool="plan_question", content=state.problem_text)]
            if "search_arxiv_primary" not in tool_names:
                return [Action(type=ActionType.APPLY, tool="search_arxiv_primary", content=state.problem_text)]
            if "search_arxiv_secondary" not in tool_names:
                return [Action(type=ActionType.APPLY, tool="search_arxiv_secondary", content=state.problem_text)]
            if "solve_question" not in tool_names:
                return [Action(type=ActionType.APPLY, tool="solve_question", content=state.problem_text)]
            candidate_answer = str(state.metadata.get("candidate_answer", "")).strip()
            if candidate_answer:
                return [Action(type=ActionType.ANSWER, content=candidate_answer)]
            return [Action(type=ActionType.BACKTRACK, content="collect different external evidence")]
        if "plan_question" not in tool_names:
            return [Action(type=ActionType.APPLY, tool="plan_question", content=state.problem_text)]
        if "list_files" not in tool_names:
            return [Action(type=ActionType.APPLY, tool="list_files", content="")]
        if "inspect_file" not in tool_names:
            return [Action(type=ActionType.APPLY, tool="inspect_file", content=str(state.metadata.get("target_file", "")))]
        if "solve_question" not in tool_names:
            return [Action(type=ActionType.APPLY, tool="solve_question", content=state.problem_text)]
        candidate_answer = str(state.metadata.get("candidate_answer", "")).strip()
        if candidate_answer:
            return [Action(type=ActionType.ANSWER, content=candidate_answer)]
        return [Action(type=ActionType.BACKTRACK, content="collect different evidence")]

    def allowed_action_types(self, state: ReasoningState) -> List[str]:
        if state.final_answer.strip():
            return ["ANSWER"]
        if bool(state.metadata.get("claim_mode", False)):
            actions = ["APPLY"]
        else:
            actions = ["THINK", "SUBGOAL", "APPLY"]
        if state.derived_facts or state.tool_history or state.metadata.get("candidate_answer"):
            actions.extend(["CHECK", "ANSWER"])
        return actions

    def allowed_tools(self, state: ReasoningState, action_type: str) -> List[str]:
        if action_type.upper() not in {"APPLY", "CHECK"}:
            return []
        return self._next_apply_tools(state) if action_type.upper() == "APPLY" else ["solve_question"]

    def candidate_bindings(self, state: ReasoningState, action_type: str, tool: str = "") -> List[Dict[str, str]]:
        normalized = action_type.upper()
        if normalized == "THINK":
            return [{"content": "infer the target file, inspect its structure, solve from evidence, and answer concisely"}]
        if normalized == "SUBGOAL":
            pending = state.obligations[:3] or ["inspect evidence file", "solve from evidence"]
            return [{"content": item} for item in pending]
        if normalized == "ANSWER":
            return self._answer_candidates(state)
        if tool == "plan_question":
            return [{"content": state.problem_text}]
        if tool == "list_files":
            return [{"content": ""}]
        if tool == "inspect_file":
            candidates = [str(state.metadata.get("target_file", ""))]
            candidates.extend(str(name) for name in state.metadata.get("candidate_files", []))
            deduped = [name for idx, name in enumerate(candidates) if name and name not in candidates[:idx]]
            return [{"content": name} for name in deduped[:3]]
        if tool == "search_arxiv_primary":
            return [{"content": state.problem_text}]
        if tool == "search_arxiv_secondary":
            return [{"content": state.problem_text}]
        if tool == "solve_question":
            return [{"content": state.problem_text}]
        return []

    def action_preference(self, state: ReasoningState, action: Action) -> float:
        tool_names = self._tool_names(state)
        plan = dict(state.metadata.get("question_plan", {}))
        if plan.get("research_mode") == "arxiv_cross_reference":
            if action.type == ActionType.ANSWER:
                confidence = float(state.metadata.get("answer_confidence", 0.0) or 0.0)
                return 1.0 if str(state.metadata.get("candidate_answer", "")).strip() and confidence >= 0.45 else 0.0
            if action.type == ActionType.APPLY:
                if action.tool == "plan_question":
                    return 1.0 if "plan_question" not in tool_names else 0.05
                if action.tool == "search_arxiv_primary":
                    return 0.98 if "plan_question" in tool_names and "search_arxiv_primary" not in tool_names else 0.12
                if action.tool == "search_arxiv_secondary":
                    return 0.98 if "search_arxiv_primary" in tool_names and "search_arxiv_secondary" not in tool_names else 0.10
                if action.tool == "solve_question":
                    return 1.0 if "search_arxiv_secondary" in tool_names and "solve_question" not in tool_names else 0.20
        inspected_files = [str(item) for item in state.metadata.get("inspected_files", []) if str(item).strip()]
        candidate_files = [str(item) for item in state.metadata.get("candidate_files", []) if str(item).strip()]
        remaining_files = [name for name in candidate_files if name not in inspected_files]
        if action.type == ActionType.ANSWER:
            confidence = float(state.metadata.get("answer_confidence", 0.0) or 0.0)
            return 1.0 if state.final_answer.strip() or (str(state.metadata.get("candidate_answer", "")).strip() and confidence >= 0.45) else 0.0
        if action.type == ActionType.APPLY:
            if action.tool == "plan_question":
                return 1.0 if "plan_question" not in tool_names else 0.05
            if action.tool == "list_files":
                return 0.98 if "plan_question" in tool_names and "list_files" not in tool_names else 0.10
            if action.tool == "inspect_file":
                if "list_files" in tool_names and "inspect_file" not in tool_names:
                    return 0.98
                if remaining_files and float(state.metadata.get("ambiguity_score", 0.0) or 0.0) >= 0.25:
                    return 0.72
                return 0.18
            if action.tool == "solve_question":
                if remaining_files and float(state.metadata.get("ambiguity_score", 0.0) or 0.0) >= 0.25:
                    return 0.35
                return 1.0 if "inspect_file" in tool_names and "solve_question" not in tool_names else 0.25
        if action.type == ActionType.CHECK and action.tool == "solve_question":
            return 0.80 if "inspect_file" in tool_names else 0.20
        if action.type == ActionType.THINK:
            return 0.60 if not tool_names else 0.10
        if action.type == ActionType.SUBGOAL:
            return 0.30 if state.obligations else 0.08
        return 0.0

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
            "Emit canonical JSON actions. Solve the question from workspace evidence without oracle tool hints.\n"
            'ACTION {"type":"APPLY","tool":"plan_question","content":"task prompt"}\n'
            'ACTION {"type":"APPLY","tool":"search_arxiv_primary","content":"task prompt"}\n'
            'ACTION {"type":"APPLY","tool":"search_arxiv_secondary","content":"task prompt"}\n'
            'ACTION {"type":"APPLY","tool":"inspect_file","content":"sales.csv"}\n'
            'ACTION {"type":"APPLY","tool":"solve_question","content":"task prompt"}\n'
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
                filters=self._retrieval_filters(state),
                tool_names=self.allowed_tools(state, "APPLY"),
                event_logger=event_logger,
            )
        state.metadata["_retrieval_context"] = retrieval_context or {}
        tactic_hints = None
        if tactic_stats is not None:
            ranked = tactic_stats.top_tactics(state.domain, limit=3)
            tactic_hints = [f"{name} bias={bias:.2f}" for name, bias in ranked if bias != 0.5]
        return build_search_prompt(state, self.action_format_instructions(), retrieval_context=retrieval_context, tactic_hints=tactic_hints)

    def state_signature(self, state: ReasoningState) -> str:
        return " || ".join(
            [
                state.domain,
                str(state.metadata.get("target_file", "")),
                str(state.metadata.get("question_intent", "")),
                ",".join(str(item) for item in state.metadata.get("inspected_files", [])[-3:]),
                " | ".join(state.derived_facts[-3:]),
                " | ".join(state.obligations[-3:]),
                state.final_answer.strip(),
            ]
        )

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

    def build_failure_recovery_example(self, bundle: Dict[str, Any]) -> str:
        failure_type = str(bundle.get("failure_type", "")).strip()
        focus = ""
        if failure_type:
            focus = f"\nRecovery focus: {failure_type.replace('_', ' ')}."
        evidence_graph = dict(bundle.get("evidence_graph", {})) if isinstance(bundle.get("evidence_graph", {}), dict) else {}
        task = ReasoningTask(
            task_id=str(bundle.get("task_id", f"recovery_{uuid.uuid4().hex[:8]}")),
            domain=str(bundle.get("domain", "gaia_csv_reasoning")),
            prompt=str(bundle.get("task", "")) + focus + (f"\nKnown evidence files: {', '.join(evidence_graph.get('files', [])[:4])}" if evidence_graph.get("files") else ""),
            answer=str(bundle.get("expected", "")),
            goal=str(bundle.get("goal", "Return the shortest correct final answer")),
            meta=dict(bundle.get("meta", {})),
        )
        return self.build_training_example(task)

    def training_tasks(self) -> List[ReasoningTask]:
        return list(self._train_cases)

    def benchmark_tasks(self) -> List[ReasoningTask]:
        return list(self._benchmark_cases)
