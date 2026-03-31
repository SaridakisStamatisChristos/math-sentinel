from __future__ import annotations

import contextlib
import contextvars
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


GaiaToolHandler = Callable[[str, Any], Dict[str, Any]]


@dataclass(frozen=True)
class GaiaOperator:
    name: str
    handler: GaiaToolHandler
    phase: str = "solve"
    description: str = ""
    supports_files: bool = False
    supports_network: bool = False
    strict_blind_allowed: bool = True
    resumable: bool = True
    mutates_workspace: bool = False

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "phase": self.phase,
            "description": self.description,
            "supports_files": self.supports_files,
            "supports_network": self.supports_network,
            "strict_blind_allowed": self.strict_blind_allowed,
            "resumable": self.resumable,
            "mutates_workspace": self.mutates_workspace,
        }


@dataclass
class GaiaCompactState:
    task_id: str
    question: str
    research_mode: str = ""
    solver_submode: str = ""
    answer_contract: str = ""
    operator_chain: List[str] = field(default_factory=list)
    route_candidates: List[str] = field(default_factory=list)
    expected_evidence_kind: str = ""
    target_file: str = ""
    candidate_files: List[str] = field(default_factory=list)
    inspected_files: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    obligations: List[str] = field(default_factory=list)
    recent_browse_events: List[str] = field(default_factory=list)
    rejected_candidates: List[str] = field(default_factory=list)
    best_candidate: str = ""
    answer_confidence: float = 0.0
    provenance: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "research_mode": self.research_mode,
            "solver_submode": self.solver_submode,
            "answer_contract": self.answer_contract,
            "operator_chain": list(self.operator_chain),
            "route_candidates": list(self.route_candidates),
            "expected_evidence_kind": self.expected_evidence_kind,
            "target_file": self.target_file,
            "candidate_files": list(self.candidate_files),
            "inspected_files": list(self.inspected_files),
            "evidence": list(self.evidence),
            "obligations": list(self.obligations),
            "recent_browse_events": list(self.recent_browse_events),
            "rejected_candidates": list(self.rejected_candidates),
            "best_candidate": self.best_candidate,
            "answer_confidence": float(self.answer_confidence),
            "provenance": list(self.provenance),
        }


def _dedupe_text(items: Iterable[Any], *, limit: int = 0) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in items:
        text = " ".join(str(item or "").split()).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
        if limit > 0 and len(ordered) >= limit:
            break
    return ordered


def _truncate(text: Any, max_chars: int) -> str:
    rendered = " ".join(str(text or "").split()).strip()
    if max_chars <= 0 or len(rendered) <= max_chars:
        return rendered
    if max_chars <= 3:
        return rendered[:max_chars]
    return rendered[: max_chars - 3].rstrip() + "..."


_ACTIVE_GAIA_CONTEXT: contextvars.ContextVar[GaiaSolveContext | None] = contextvars.ContextVar(
    "active_gaia_context",
    default=None,
)


@dataclass
class GaiaSolveContext:
    task_id: str
    prompt: str
    workspace_dir: str
    available_files: List[str]
    metadata: Dict[str, Any]
    question_plan: Dict[str, Any] = field(default_factory=dict)
    progress_log_path: str = ""
    resume_snapshot_path: str = ""
    operator_names: List[str] = field(default_factory=list)
    stage: str = ""
    resume_enabled: bool = False
    started_at: float = field(default_factory=time.time)
    progress_events: List[Dict[str, Any]] = field(default_factory=list)
    recent_candidates: List[Dict[str, Any]] = field(default_factory=list)

    @contextlib.contextmanager
    def activate(self) -> Iterable[GaiaSolveContext]:
        token = _ACTIVE_GAIA_CONTEXT.set(self)
        try:
            yield self
        finally:
            _ACTIVE_GAIA_CONTEXT.reset(token)

    def emit(self, event: str, **payload: Any) -> None:
        record: Dict[str, Any] = {
            "ts": round(time.time(), 3),
            "event": str(event or "").strip(),
            "task_id": self.task_id,
            "stage": self.stage,
        }
        for key, value in payload.items():
            if value is None:
                continue
            record[str(key)] = value
        self.progress_events.append(record)
        if len(self.progress_events) > 64:
            self.progress_events = self.progress_events[-64:]
        if self.progress_log_path:
            path = Path(self.progress_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def remember_candidate(
        self,
        candidate: str,
        *,
        accepted: bool,
        score: float | None = None,
        notes: Sequence[str] = (),
        method: str = "",
    ) -> None:
        entry: Dict[str, Any] = {
            "candidate": " ".join(str(candidate or "").split()).strip(),
            "accepted": bool(accepted),
        }
        if score is not None:
            entry["score"] = float(score)
        if method:
            entry["method"] = str(method)
        cleaned_notes = _dedupe_text(notes, limit=4)
        if cleaned_notes:
            entry["notes"] = cleaned_notes
        self.recent_candidates.append(entry)
        if len(self.recent_candidates) > 12:
            self.recent_candidates = self.recent_candidates[-12:]

    def recent_progress(self, *, limit: int = 8, text_item_chars: int = 140) -> List[str]:
        rendered: List[str] = []
        for item in self.progress_events[-limit:]:
            event = str(item.get("event", "")).strip()
            if not event:
                continue
            detail_parts: List[str] = []
            for key in ("query", "url", "mode", "status", "count", "candidate", "reason"):
                value = item.get(key)
                if value in (None, ""):
                    continue
                detail_parts.append(f"{key}={_truncate(value, text_item_chars)}")
            suffix = f" ({'; '.join(detail_parts)})" if detail_parts else ""
            rendered.append(f"{event}{suffix}")
        return rendered

    def load_resume_snapshot(self) -> Dict[str, Any]:
        if not self.resume_enabled or not self.resume_snapshot_path:
            return {}
        path = Path(self.resume_snapshot_path)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_resume_snapshot(
        self,
        compact_state: Mapping[str, Any],
        *,
        last_result: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not self.resume_snapshot_path:
            return
        snapshot = {
            "task_id": self.task_id,
            "stage": self.stage,
            "prompt": self.prompt,
            "question_plan": dict(self.question_plan),
            "metadata": {
                "target_file": self.metadata.get("target_file", ""),
                "candidate_files": list(self.metadata.get("candidate_files", []) or []),
                "inspected_files": list(self.metadata.get("inspected_files", []) or []),
            },
            "compact_state": dict(compact_state),
            "recent_progress": self.recent_progress(limit=10),
            "recent_candidates": list(self.recent_candidates[-8:]),
            "saved_at": round(time.time(), 3),
        }
        if last_result:
            snapshot["last_result"] = {
                "ok": bool(last_result.get("ok", False)),
                "result": str(last_result.get("result", "") or ""),
                "solved": bool(last_result.get("solved", False)),
                "answer": str(last_result.get("answer", "") or ""),
                "risk": float(last_result.get("risk", 0.0) or 0.0),
            }
        path = Path(self.resume_snapshot_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, ensure_ascii=True, indent=2), encoding="utf-8")


def get_active_gaia_context() -> GaiaSolveContext | None:
    return _ACTIVE_GAIA_CONTEXT.get()


class GaiaQueryEngine:
    def __init__(self, operators: Mapping[str, GaiaOperator]) -> None:
        self.operators = dict(operators)

    def operator_metadata(self) -> List[Dict[str, Any]]:
        return [self.operators[name].to_metadata() for name in sorted(self.operators)]

    def run_stage(
        self,
        stage: str,
        context: GaiaSolveContext,
        callback: Callable[[GaiaSolveContext], Dict[str, Any]],
        *,
        compact_state: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        context.stage = stage
        context.emit(
            f"{stage}_start",
            available_files=len(context.available_files),
            research_mode=str(context.question_plan.get("research_mode", "") or ""),
        )
        with context.activate():
            result = callback(context)
        context.emit(
            f"{stage}_finish",
            ok=bool(result.get("ok", False)),
            solved=bool(result.get("solved", False)),
            risk=float(result.get("risk", 0.0) or 0.0),
        )
        return self.finalize_result(stage, context, result, compact_state=compact_state)

    def finalize_result(
        self,
        stage: str,
        context: GaiaSolveContext,
        result: Dict[str, Any],
        *,
        compact_state: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = dict(result.get("payload", result.get("result_payload", {})) or {})
        state_metadata = dict(payload.get("state_metadata", {}) or {})
        state_metadata["gaia_progress_log_path"] = context.progress_log_path
        state_metadata["gaia_resume_snapshot_path"] = context.resume_snapshot_path
        state_metadata["gaia_runtime_stage"] = stage
        state_metadata["gaia_recent_progress"] = context.recent_progress(limit=8)
        state_metadata["gaia_recent_candidates"] = list(context.recent_candidates[-8:])
        state_metadata["gaia_operator_registry"] = self.operator_metadata()
        if compact_state is not None:
            state_metadata["gaia_compact_state"] = dict(compact_state)
        payload["state_metadata"] = state_metadata
        result["payload"] = payload
        if compact_state is not None:
            context.save_resume_snapshot(compact_state, last_result=result)
        return result
