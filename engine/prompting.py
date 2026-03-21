from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from .state import ReasoningState


DEFAULT_PROMPT_COMPACTION: Dict[str, int | bool] = {
    "enabled": False,
    "problem_chars": 720,
    "fact_limit": 4,
    "subgoal_limit": 3,
    "obligation_limit": 4,
    "evidence_limit": 4,
    "tool_limit": 3,
    "action_limit": 2,
    "file_limit": 5,
    "retrieval_item_limit": 2,
    "text_item_chars": 120,
}


def _truncate(text: Any, max_chars: int) -> str:
    rendered = re.sub(r"\s+", " ", str(text or "")).strip()
    if max_chars <= 0 or len(rendered) <= max_chars:
        return rendered
    if max_chars <= 3:
        return rendered[:max_chars]
    keep = max_chars - 3
    head = max(1, int(keep * 0.65))
    tail = max(1, keep - head)
    return rendered[:head].rstrip() + "..." + rendered[-tail:].lstrip()


def _dedupe_preserve(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _compact_problem(problem_text: str, max_chars: int) -> str:
    rendered = str(problem_text or "").strip()
    if len(rendered) <= max_chars:
        return rendered

    task_body, _, _ = rendered.partition("\nWorkspace files:\n")
    lines = [line.strip() for line in re.split(r"[\r\n]+", task_body) if line.strip()]
    if not lines:
        return _truncate(rendered, max_chars)

    scored: List[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        lowered = line.lower()
        score = 0
        if index == 0:
            score += 8
        if re.search(r"https?://", line):
            score += 5
        if re.search(r"\b(19|20)\d{2}\b", line):
            score += 3
        if re.search(r"[\"“].+?[\"”]", line):
            score += 3
        if any(
            token in lowered
            for token in (
                "answer",
                "return",
                "format",
                "give",
                "using",
                "use ",
                "rounded",
                "round",
                "separated",
                "list",
                "only",
                "include",
                "exclude",
                "before",
                "after",
                "between",
                "closest",
                "latest",
                "earliest",
                "oldest",
                "newest",
                "distance",
                "how many",
                "what is",
                "which",
            )
        ):
            score += 4
        scored.append((score, index, line))

    selected_indexes = {0, len(lines) - 1}
    for _, index, _ in sorted(scored, key=lambda item: (-item[0], item[1])):
        selected_indexes.add(index)
        chosen = " ".join(lines[i] for i in sorted(selected_indexes))
        if len(chosen) >= max_chars:
            break

    return _truncate(" ".join(lines[i] for i in sorted(selected_indexes)), max_chars)


def _string_items(items: Iterable[Any], limit: int, text_item_chars: int, *, from_end: bool = False) -> List[str]:
    values = [item for item in items if str(item or "").strip()]
    if from_end:
        values = values[-limit:]
    else:
        values = values[:limit]
    return [_truncate(item, text_item_chars) for item in values]


def _summarize_tool_history(state: ReasoningState, limit: int, text_item_chars: int) -> List[str]:
    summaries: List[str] = []
    for record in list(getattr(state, "tool_history", []))[-limit:]:
        if not isinstance(record, dict):
            continue
        tool = str(record.get("tool", "")).strip() or "tool"
        result = record.get("result", {}) if isinstance(record.get("result", {}), dict) else {}
        payload = result.get("payload", {}) if isinstance(result.get("payload", {}), dict) else {}
        answer = (
            result.get("answer")
            or result.get("result")
            or payload.get("candidate_answer")
            or payload.get("answer")
            or ""
        )
        progress = result.get("goal_progress")
        parts = [tool]
        if str(answer).strip():
            parts.append(f"-> {_truncate(answer, text_item_chars)}")
        if progress is not None:
            try:
                parts.append(f"(progress={float(progress):.2f})")
            except Exception:
                pass
        summaries.append(" ".join(parts).strip())
    return summaries


def _summarize_action_history(state: ReasoningState, limit: int, text_item_chars: int) -> List[str]:
    summaries: List[str] = []
    for record in list(getattr(state, "action_history", []))[-limit:]:
        if not isinstance(record, dict):
            continue
        action_type = str(record.get("type", "")).strip() or "ACTION"
        tool = str(record.get("tool", "")).strip()
        content = _truncate(record.get("content", ""), text_item_chars)
        if tool:
            summaries.append(f"{action_type}:{tool} {content}".strip())
        else:
            summaries.append(f"{action_type} {content}".strip())
    return summaries


def _compact_metadata(state: ReasoningState, file_limit: int, text_item_chars: int) -> List[str]:
    metadata = getattr(state, "metadata", {}) or {}
    entries: List[str] = []
    target_file = str(metadata.get("target_file", "")).strip()
    if target_file:
        entries.append(f"target_file={target_file}")
    candidate_files = [str(item) for item in metadata.get("candidate_files", []) if str(item).strip()]
    if candidate_files:
        entries.append("candidate_files=" + ", ".join(candidate_files[:file_limit]))
    inspected = [str(item) for item in metadata.get("inspected_files", []) if str(item).strip()]
    if inspected:
        entries.append("inspected=" + ", ".join(inspected[-file_limit:]))
    research_mode = str(metadata.get("research_mode", "")).strip()
    if research_mode:
        entries.append(f"research_mode={research_mode}")
    question_intent = str(metadata.get("question_intent", "")).strip()
    if question_intent:
        entries.append(f"intent={_truncate(question_intent, text_item_chars)}")
    candidate_answer = str(metadata.get("candidate_answer", "")).strip()
    if candidate_answer:
        entries.append(f"candidate_answer={_truncate(candidate_answer, text_item_chars)}")
    answer_confidence = metadata.get("answer_confidence", None)
    if answer_confidence is not None:
        try:
            entries.append(f"answer_confidence={float(answer_confidence):.2f}")
        except Exception:
            pass
    return entries


def _prompt_compaction_options(state: ReasoningState) -> Dict[str, int | bool]:
    options: Dict[str, int | bool] = dict(DEFAULT_PROMPT_COMPACTION)
    metadata = getattr(state, "metadata", {}) or {}
    raw = metadata.get("prompt_compaction", {})
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key in options:
                options[key] = value
    return options


def _is_generic_goal(goal: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(goal or "")).strip().lower()
    return normalized in {
        "",
        "return the answer",
        "return only the answer",
        "return only the final answer",
        "return the shortest correct answer",
        "return the shortest correct final answer",
        "return the correct final answer",
    }


def _render_search_state(state: ReasoningState) -> str:
    options = _prompt_compaction_options(state)
    if not bool(options.get("enabled", False)):
        return state.serialize()

    text_item_chars = int(options.get("text_item_chars", 120))
    lines = [
        f"[DOMAIN] {state.domain}",
        f"[TASK] {_compact_problem(state.problem_text, int(options.get('problem_chars', 720)))}",
    ]
    if not _is_generic_goal(state.goal):
        lines.append(f"[GOAL] {_truncate(state.goal, text_item_chars)}")

    metadata_lines = _compact_metadata(state, int(options.get("file_limit", 5)), text_item_chars)
    if metadata_lines:
        lines.append(f"[FOCUS] {' | '.join(metadata_lines)}")

    assumptions = _string_items(state.assumptions, int(options.get("subgoal_limit", 3)), text_item_chars)
    if assumptions:
        lines.append(f"[ASSUMPTIONS] {' | '.join(assumptions)}")
    derived = _string_items(state.derived_facts, int(options.get("fact_limit", 4)), text_item_chars, from_end=True)
    if derived:
        lines.append(f"[DERIVED] {' | '.join(derived)}")
    subgoals = _string_items(state.subgoals, int(options.get("subgoal_limit", 3)), text_item_chars)
    if subgoals:
        lines.append(f"[SUBGOALS] {' | '.join(subgoals)}")
    obligations = _string_items(state.obligations, int(options.get("obligation_limit", 4)), text_item_chars)
    if obligations:
        lines.append(f"[OBLIGATIONS] {' | '.join(obligations)}")
    evidence = _string_items(state.evidence_refs, int(options.get("evidence_limit", 4)), text_item_chars, from_end=True)
    if evidence:
        lines.append(f"[EVIDENCE] {' | '.join(evidence)}")

    tool_history = _summarize_tool_history(state, int(options.get("tool_limit", 3)), text_item_chars)
    if tool_history:
        lines.append(f"[RECENT_TOOLS] {' | '.join(tool_history)}")
    action_history = _summarize_action_history(state, int(options.get("action_limit", 2)), text_item_chars)
    if action_history:
        lines.append(f"[RECENT_ACTIONS] {' | '.join(action_history)}")

    if state.status != "open" or state.final_answer.strip():
        lines.append(f"[STATUS] {state.status}")
    if state.final_answer.strip():
        lines.append(f"[FINAL_ANSWER] {_truncate(state.final_answer, text_item_chars)}")
    if float(state.terminal_confidence) > 0.0:
        lines.append(f"[TERMINAL_CONFIDENCE] {state.terminal_confidence:.3f}")
    lines.append("[END_STATE]")
    return "\n".join(lines)


def _render_lemmas(lemmas: Iterable[Any], *, limit: int = 3, text_item_chars: int = 120) -> str:
    lines = []
    for lemma in list(lemmas)[:limit]:
        name = getattr(lemma, "name", "lemma")
        pattern = _truncate(getattr(lemma, "pattern", ""), text_item_chars)
        chain = getattr(lemma, "tactic_chain", [])
        chain_text = _truncate(" -> ".join(chain) if isinstance(chain, list) else str(chain), text_item_chars)
        lines.append(f"- {name}: {pattern} | tactics={chain_text}")
    return "\n".join(lines) if lines else "none"


def _render_hard_cases(cases: Iterable[Dict[str, Any]], *, limit: int = 3, text_item_chars: int = 120) -> str:
    lines = []
    for case in list(cases)[:limit]:
        task = _truncate(case.get("task", ""), text_item_chars)
        answer = _truncate(case.get("answer", ""), text_item_chars)
        expected = _truncate(case.get("expected", ""), text_item_chars)
        lines.append(f"- task={task} | answer={answer} | expected={expected}")
    return "\n".join(lines) if lines else "none"


def build_search_prompt(
    state: ReasoningState,
    action_instructions: str,
    *,
    retrieval_context: Optional[Dict[str, list[Any]]] = None,
    tactic_hints: Optional[list[str]] = None,
) -> str:
    options = _prompt_compaction_options(state)
    retrieval_item_limit = int(options.get("retrieval_item_limit", 2))
    text_item_chars = int(options.get("text_item_chars", 120))
    sections = [_render_search_state(state)]
    if retrieval_context:
        sections.append("[RETRIEVED_LEMMAS]")
        sections.append(
            _render_lemmas(
                retrieval_context.get("lemmas", []),
                limit=retrieval_item_limit,
                text_item_chars=text_item_chars,
            )
        )
        sections.append("[SIMILAR_HARD_CASES]")
        sections.append(
            _render_hard_cases(
                retrieval_context.get("hard_cases", []),
                limit=retrieval_item_limit,
                text_item_chars=text_item_chars,
            )
        )
        tool_priors = retrieval_context.get("tool_priors", {})
        if tool_priors:
            sections.append("[RETRIEVAL_TOOL_HINTS]")
            tool_lines = [
                f"- {tool}: {score:.2f}"
                for tool, score in sorted(tool_priors.items(), key=lambda item: item[1], reverse=True)[:retrieval_item_limit + 1]
            ]
            sections.append("\n".join(tool_lines))
        failure_avoidance = retrieval_context.get("failure_avoidance", [])
        if failure_avoidance:
            sections.append("[FAILURE_AVOIDANCE]")
            sections.append("\n".join(f"- {_truncate(item, text_item_chars)}" for item in list(failure_avoidance)[:retrieval_item_limit]))
    if tactic_hints:
        sections.append("[TACTIC_HINTS]")
        sections.append("\n".join(f"- {_truncate(hint, text_item_chars)}" for hint in list(tactic_hints)[:retrieval_item_limit + 1]))
    sections.append(action_instructions.strip())
    return "\n".join(sections)
