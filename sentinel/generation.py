
from __future__ import annotations

import math
from typing import Any, Iterable, List, Sequence

import torch
import torch.nn.functional as F

from engine.action_format import render_canonical_action
from engine.actions import Action, ActionType
from .model import TinyTransformerLM
from .tokenizer import StructuredTokenizer


def generate_text(
    model: TinyTransformerLM,
    tokenizer: StructuredTokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 24,
) -> str:
    ids = tokenizer.encode(prompt, model.seq_len)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    valid = (x[0] != tokenizer.pad_id).nonzero(as_tuple=False)
    last = int(valid[-1].item()) + 1 if len(valid) else 1
    x = x[:, :last]
    out = model.generate_ids(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
    return tokenizer.decode(out[0].tolist())


def _unique_texts(texts: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for text in texts:
        normalized = text.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _recent_tool_results(state: Any | None) -> List[str]:
    if state is None:
        return []
    values: List[str] = []
    for record in getattr(state, "tool_history", [])[-4:]:
        result = record.get("result", {}) if isinstance(record, dict) else {}
        rendered = result.get("answer") or result.get("result") or ""
        text = str(rendered).strip()
        if text:
            values.append(text)
    return values


def _content_candidates(state: Any | None, action_type: ActionType) -> List[str]:
    if state is None:
        base = ["plan next step"]
        return base if action_type == ActionType.THINK else []

    goal = str(getattr(state, "goal", "")).strip()
    problem_text = str(getattr(state, "problem_text", "")).strip()
    derived = [str(item).strip() for item in getattr(state, "derived_facts", [])[-4:] if str(item).strip()]
    subgoals = [str(item).strip() for item in getattr(state, "subgoals", [])[-4:] if str(item).strip()]
    tool_results = _recent_tool_results(state)

    if action_type == ActionType.THINK:
        return _unique_texts(
            [
                "plan next grounded step",
                "inspect tools and derived facts",
                f"work toward goal: {goal}" if goal else "",
            ]
        )
    if action_type == ActionType.SUBGOAL:
        return _unique_texts(subgoals + ([goal] if goal else []) + ["derive a usable intermediate result"])
    if action_type == ActionType.RESOLVE_SUBGOAL:
        return _unique_texts(subgoals)
    if action_type == ActionType.ANSWER:
        return _unique_texts(tool_results + derived + [str(getattr(state, "final_answer", "")).strip()])
    if action_type in {ActionType.APPLY, ActionType.CHECK, ActionType.REWRITE, ActionType.SIMPLIFY, ActionType.CALL_PLUGIN}:
        return _unique_texts([problem_text] + derived + subgoals + tool_results)
    if action_type == ActionType.LEMMA:
        return _unique_texts(derived[:2] + subgoals[:1])
    if action_type == ActionType.ASSUME:
        return _unique_texts([goal] + derived[:2])
    if action_type == ActionType.BACKTRACK:
        return ["search dead end"]
    return []


def _candidate_action_types(state: Any | None) -> List[ActionType]:
    if state is None:
        return [ActionType.THINK, ActionType.APPLY]

    types: List[ActionType] = [ActionType.THINK, ActionType.APPLY, ActionType.SUBGOAL]
    if getattr(state, "subgoals", []):
        types.append(ActionType.RESOLVE_SUBGOAL)
    if getattr(state, "derived_facts", []) or getattr(state, "tool_history", []):
        types.extend([ActionType.CHECK, ActionType.SIMPLIFY, ActionType.ANSWER])
    if getattr(state, "action_history", []):
        types.append(ActionType.BACKTRACK)
    return types


def _rank_tools(state: Any | None, available_tools: Sequence[str] | None) -> List[str]:
    tools = list(available_tools or [])
    if not tools:
        return []
    if state is None:
        return tools[:6]

    text = f"{getattr(state, 'domain', '')} {getattr(state, 'problem_text', '')}".lower()

    def _score(tool: str) -> tuple[int, int, str]:
        parts = tool.lower().split("_")
        exact = int(tool.lower() in text)
        overlap = sum(int(part and part in text) for part in parts)
        return (-exact, -overlap, tool)

    ranked = sorted(tools, key=_score)
    return ranked[:8]


def _schema_candidates(state: Any | None, schema_provider: Any | None) -> List[str]:
    if state is None or schema_provider is None:
        return []
    required = ("allowed_action_types", "allowed_tools", "candidate_bindings")
    if not all(hasattr(schema_provider, name) for name in required):
        return []

    candidates: List[str] = []
    for raw_action_type in schema_provider.allowed_action_types(state):
        action_name = str(raw_action_type).upper()
        try:
            action_type = ActionType(action_name)
        except Exception:
            continue
        if action_type in {ActionType.APPLY, ActionType.CHECK, ActionType.REWRITE, ActionType.SIMPLIFY, ActionType.CALL_PLUGIN}:
            for tool in schema_provider.allowed_tools(state, action_type.value):
                for binding in schema_provider.candidate_bindings(state, action_type.value, tool):
                    content = str(binding.get("content", "")).strip()
                    action = Action(type=action_type, tool=str(tool), content=content)
                    if action.validate():
                        candidates.append(render_canonical_action(action))
        else:
            for binding in schema_provider.candidate_bindings(state, action_type.value, ""):
                content = str(binding.get("content", "")).strip()
                action = Action(type=action_type, content=content)
                if action.validate():
                    candidates.append(render_canonical_action(action))
    return _unique_texts(candidates)


def build_structured_action_candidates(
    state: Any | None,
    available_tools: Sequence[str] | None = None,
    schema_provider: Any | None = None,
) -> List[str]:
    schema_first = _schema_candidates(state, schema_provider)
    if schema_first:
        return schema_first
    candidates: List[str] = []
    ranked_tools = _rank_tools(state, available_tools)
    for action_type in _candidate_action_types(state):
        contents = _content_candidates(state, action_type)
        if action_type in {ActionType.APPLY, ActionType.CHECK, ActionType.REWRITE, ActionType.SIMPLIFY, ActionType.CALL_PLUGIN}:
            for tool in ranked_tools:
                for content in contents[:3]:
                    action = Action(type=action_type, tool=tool, content=content)
                    if action.validate():
                        candidates.append(render_canonical_action(action))
        else:
            for content in contents[:3]:
                action = Action(type=action_type, content=content)
                if action.validate():
                    candidates.append(render_canonical_action(action))
    return _unique_texts(candidates)


def _trim_prefix_ids(prefix_ids: List[int], continuation_ids: List[int], seq_len: int) -> tuple[List[int], int]:
    continuation_ids = continuation_ids[: max(1, seq_len - 1)]
    max_prefix = max(1, seq_len - len(continuation_ids))
    trimmed_prefix = prefix_ids[-max_prefix:]
    return trimmed_prefix + continuation_ids, len(trimmed_prefix)


def _score_candidate_texts(
    model: TinyTransformerLM,
    tokenizer: StructuredTokenizer,
    prompt: str,
    candidates: Sequence[str],
    device: str,
) -> List[float]:
    if not candidates:
        return []

    prefix_ids = tokenizer.encode_unpadded(prompt, add_bos=True, add_eos=False)
    sequences: List[List[int]] = []
    starts: List[int] = []
    lengths: List[int] = []
    for candidate in candidates:
        continuation_ids = tokenizer.encode_unpadded(candidate, add_bos=False, add_eos=False)
        full_ids, start = _trim_prefix_ids(prefix_ids, continuation_ids, model.seq_len)
        sequences.append(full_ids)
        starts.append(start)
        lengths.append(len(full_ids) - start)

    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [tokenizer.pad_id] * (max_len - len(seq)) for seq in sequences]
    x = torch.tensor(padded, dtype=torch.long, device=device)
    logits = model(x[:, :-1])
    log_probs = F.log_softmax(logits, dim=-1)
    targets = x[:, 1:]

    scores: List[float] = []
    for idx, (start, length) in enumerate(zip(starts, lengths)):
        start_index = max(0, start - 1)
        end_index = min(start_index + length, targets.shape[1])
        if end_index <= start_index:
            scores.append(float("-inf"))
            continue
        target_slice = targets[idx, start_index:end_index]
        log_prob_slice = log_probs[idx, start_index:end_index]
        gathered = log_prob_slice.gather(1, target_slice.unsqueeze(1)).squeeze(1)
        scores.append(float(gathered.mean().item()))
    return scores


def propose_structured_actions(
    model: TinyTransformerLM,
    tokenizer: StructuredTokenizer,
    prompt: str,
    device: str,
    *,
    state: Any | None = None,
    available_tools: Sequence[str] | None = None,
    proposal_count: int = 4,
    temperature: float = 0.8,
    schema_provider: Any | None = None,
) -> List[str]:
    if not hasattr(tokenizer, "encode_unpadded") or not hasattr(tokenizer, "tokenize"):
        return []

    candidates = build_structured_action_candidates(state, available_tools=available_tools, schema_provider=schema_provider)
    if not candidates:
        return []

    scores = _score_candidate_texts(model, tokenizer, prompt, candidates, device)
    ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
    if temperature > 0 and len(ranked) > proposal_count:
        top = ranked[: max(proposal_count * 2, proposal_count)]
        weight_values = torch.tensor([score for _, score in top], dtype=torch.float32)
        probs = torch.softmax(weight_values / max(temperature, 1e-4), dim=0)
        picked: List[str] = []
        picked_idx: set[int] = set()
        while len(picked) < proposal_count and len(picked_idx) < len(top):
            next_idx = int(torch.multinomial(probs, num_samples=1).item())
            if next_idx in picked_idx:
                probs[next_idx] = 0.0
                if float(probs.sum()) > 0:
                    probs = probs / probs.sum()
                continue
            picked_idx.add(next_idx)
            picked.append(top[next_idx][0])
        if picked:
            return picked
    return [text for text, _ in ranked[:proposal_count]]


def propose_actions(
    model: TinyTransformerLM,
    tokenizer: StructuredTokenizer,
    prompt: str,
    device: str,
    proposal_count: int = 4,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 24,
    state: Any | None = None,
    available_tools: Sequence[str] | None = None,
    decoder_mode: str = "hybrid",
    schema_provider: Any | None = None,
) -> List[str]:
    proposals: List[str] = []
    mode = (decoder_mode or "hybrid").lower()
    if mode == "legacy_text":
        mode = "free"
    if mode == "relaxed":
        mode = "hybrid"
    structured_target = proposal_count if mode in {"strict", "structured"} else max(1, proposal_count // 2)
    if mode in {"strict", "structured", "hybrid"}:
        proposals.extend(
            propose_structured_actions(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                state=state,
                available_tools=available_tools,
                proposal_count=structured_target,
                temperature=temperature,
                schema_provider=schema_provider,
            )
        )

    if mode in {"free", "hybrid"} and len(proposals) < proposal_count:
        for _ in range(proposal_count - len(proposals)):
            proposals.append(
                generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
            )
    return _unique_texts(proposals)[:proposal_count]
