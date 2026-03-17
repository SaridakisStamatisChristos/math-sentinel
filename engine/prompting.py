from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .state import ReasoningState


def _render_lemmas(lemmas: Iterable[Any]) -> str:
    lines = []
    for lemma in lemmas:
        name = getattr(lemma, "name", "lemma")
        pattern = getattr(lemma, "pattern", "")
        chain = getattr(lemma, "tactic_chain", [])
        chain_text = " -> ".join(chain) if isinstance(chain, list) else str(chain)
        lines.append(f"- {name}: {pattern} | tactics={chain_text}")
    return "\n".join(lines) if lines else "none"


def _render_hard_cases(cases: Iterable[Dict[str, Any]]) -> str:
    lines = []
    for case in cases:
        task = str(case.get("task", ""))[:120]
        answer = case.get("answer", "")
        expected = case.get("expected", "")
        lines.append(f"- task={task} | answer={answer} | expected={expected}")
    return "\n".join(lines) if lines else "none"


def build_search_prompt(
    state: ReasoningState,
    action_instructions: str,
    *,
    retrieval_context: Optional[Dict[str, list[Any]]] = None,
    tactic_hints: Optional[list[str]] = None,
) -> str:
    sections = [state.serialize()]
    if retrieval_context:
        sections.append("[RETRIEVED_LEMMAS]")
        sections.append(_render_lemmas(retrieval_context.get("lemmas", [])))
        sections.append("[SIMILAR_HARD_CASES]")
        sections.append(_render_hard_cases(retrieval_context.get("hard_cases", [])))
    if tactic_hints:
        sections.append("[TACTIC_HINTS]")
        sections.append("\n".join(f"- {hint}" for hint in tactic_hints))
    sections.append(action_instructions.strip())
    return "\n".join(sections)
