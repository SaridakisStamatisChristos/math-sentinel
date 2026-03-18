
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from proof.parser import parse_actions
from proof.state import ProofState
from proof.executor import ProofExecutor
from sentinel.generation import propose_actions
from sentinel.tokenizer import StructuredTokenizer
from sentinel.model import TinyTransformerLM
from sentinel.verifier import StateVerifier

from .nodes import SearchNode
from .repair import fallback_repairs
from .scoring import combine_scores
from .transposition import TranspositionTable


def _build_prompt(state: ProofState) -> str:
    instruction = (
        "\nPrefer the canonical action format with one JSON action per line, for example:\n"
        'ACTION {"type":"THINK","content":"plan"}\n'
        'ACTION {"type":"APPLY","tool":"tool_name","content":"arguments"}\n'
        'ACTION {"type":"ANSWER","content":"final answer"}\n'
        "Legacy XML-style <action ...>...</action> output is still accepted.\n"
        "Always finish with an ANSWER action once you have enough information. "
        "Do not stop after THINK or APPLY if a final answer or proof sketch is available.\n"
    )
    return state.serialize() + instruction


def _default_state_signature(state: ProofState) -> str:
    return " || ".join(
        [
            state.domain,
            state.problem_text,
            " | ".join(state.assumptions[-3:]),
            " | ".join(state.derived_facts[-3:]),
            " | ".join(state.subgoals[-3:]),
            state.status,
            state.final_answer.strip(),
        ]
    )


def _score_state(verifier: StateVerifier, tokenizer: StructuredTokenizer, device: str, state: ProofState) -> Dict[str, float]:
    ids = tokenizer.encode(state.serialize(), seq_len=384)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        pred = verifier.predict_scores(x)
    return {k: float(v.item()) for k, v in pred.items()}


def beam_search(
    prover: TinyTransformerLM,
    verifier: StateVerifier,
    tokenizer: StructuredTokenizer,
    executor: ProofExecutor,
    initial_state: ProofState,
    device: str,
    beam_width: int = 4,
    max_depth: int = 4,
    proposal_count: int = 4,
    max_new_tokens: int = 72,
    temperature: float = 0.8,
    top_k: int = 24,
    score_config: Optional[Dict[str, Any]] = None,
    parse_actions_fn: Callable[[str], Tuple[List[Any], float]] = parse_actions,
    fallback_repairs_fn: Callable[[ProofState], List[Any]] = fallback_repairs,
    prompt_builder: Callable[[ProofState], str] = _build_prompt,
    state_signature_fn: Callable[[ProofState], str] = _default_state_signature,
    action_bias_fn: Optional[Callable[[ProofState, Any], float]] = None,
    schema_provider: Any | None = None,
    event_logger: Optional[Callable[..., Any]] = None,
) -> Tuple[ProofState, List[SearchNode]]:
    score_config = score_config or {}
    root = SearchNode(state=initial_state.clone(), cumulative_score=0.0, depth=0)
    beam: List[SearchNode] = [root]
    explored: List[SearchNode] = [root]
    best = root
    transpositions = TranspositionTable(capacity=int(score_config.get("transposition_capacity", 4096)))
    transpositions.register(state_signature_fn(root.state), 0.0, 0)
    available_tools = tuple(getattr(getattr(executor, "tool_registry", None), "tools", {}).keys())
    decoder_mode = str(score_config.get("decoder_mode", "hybrid"))

    for depth in range(1, max_depth + 1):
        candidates: List[SearchNode] = []
        for node in beam:
            if node.state.status == "solved":
                candidates.append(node)
                continue

            prompt = prompt_builder(node.state)
            texts = propose_actions(
                model=prover,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                proposal_count=proposal_count,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                state=node.state,
                available_tools=available_tools,
                decoder_mode=decoder_mode,
                schema_provider=schema_provider,
            )

            for txt in texts:
                actions, confidence = parse_actions_fn(txt)
                if not actions:
                    if event_logger is not None:
                        event_logger("invalid_schema_emission", domain=node.state.domain, depth=depth, text=txt[:200])
                    continue
                state = node.state
                exec_info: Dict[str, float] = {"goal_progress": 0.0, "valid_step": confidence}
                child_state = state
                last_action = None
                for action in actions:
                    child_state, local = executor.apply(child_state, action)
                    exec_info["goal_progress"] = exec_info.get("goal_progress", 0.0) + float(local.get("goal_progress", 0.0))
                    exec_info["valid_step"] = min(exec_info.get("valid_step", 1.0), float(local.get("valid_step", 1.0)))
                    last_action = action
                    if float(local.get("valid_step", 1.0)) <= 0.0 and event_logger is not None:
                        event_logger(
                            "tool_failure",
                            domain=node.state.domain,
                            depth=depth,
                            action=action.type.value,
                            tool=action.tool,
                            note=str(local.get("note", ""))[:200],
                        )
                    if child_state.status == "solved":
                        break
                exec_info["answer_present"] = 1.0 if child_state.final_answer.strip() else 0.0
                exec_info["tactic_bias"] = action_bias_fn(node.state, last_action) if action_bias_fn and last_action is not None else 0.5
                exec_info["novelty_bonus"] = 1.0
                verifier_scores = _score_state(verifier, tokenizer, device, child_state)
                score = node.cumulative_score + combine_scores(
                    verifier_scores,
                    exec_info,
                    simplicity_penalty=float(score_config.get("simplicity_penalty", 0.02)),
                    invalid_penalty=float(score_config.get("invalid_penalty", 1.0)),
                    goal_bonus=float(score_config.get("goal_bonus", 0.4)),
                    solved_bonus=float(score_config.get("solved_bonus", 1.0)),
                    completion_bonus=float(score_config.get("completion_bonus", 0.15)),
                    incomplete_penalty=float(score_config.get("incomplete_penalty", 0.35)),
                    tactic_bonus=float(score_config.get("tactic_bonus", 0.12)),
                    value_weight=float(score_config.get("value_weight", 0.35)),
                    novelty_weight=float(score_config.get("novelty_weight", 0.08)),
                    depth=depth,
                    solved=(child_state.status == "solved"),
                )
                signature = state_signature_fn(child_state)
                accepted, novelty = transpositions.register(signature, score, depth)
                if not accepted:
                    continue
                if novelty != 1.0:
                    exec_info["novelty_bonus"] = novelty
                    score = node.cumulative_score + combine_scores(
                        verifier_scores,
                        exec_info,
                        simplicity_penalty=float(score_config.get("simplicity_penalty", 0.02)),
                        invalid_penalty=float(score_config.get("invalid_penalty", 1.0)),
                        goal_bonus=float(score_config.get("goal_bonus", 0.4)),
                        solved_bonus=float(score_config.get("solved_bonus", 1.0)),
                        completion_bonus=float(score_config.get("completion_bonus", 0.15)),
                        incomplete_penalty=float(score_config.get("incomplete_penalty", 0.35)),
                        tactic_bonus=float(score_config.get("tactic_bonus", 0.12)),
                        value_weight=float(score_config.get("value_weight", 0.35)),
                        novelty_weight=float(score_config.get("novelty_weight", 0.08)),
                        depth=depth,
                        solved=(child_state.status == "solved"),
                    )
                if event_logger is not None and abs(float(verifier_scores.get("branch_priority", 0.0)) - float(verifier_scores.get("value_estimate", 0.0))) > 0.4:
                    event_logger(
                        "verifier_value_disagreement",
                        domain=node.state.domain,
                        depth=depth,
                        branch_priority=float(verifier_scores.get("branch_priority", 0.0)),
                        value_estimate=float(verifier_scores.get("value_estimate", 0.0)),
                    )
                child = SearchNode(
                    state=child_state,
                    cumulative_score=score,
                    local_scores={**verifier_scores, **exec_info},
                    parent=node,
                    action=last_action,
                    depth=depth,
                )
                candidates.append(child)
                explored.append(child)

            for repair in fallback_repairs_fn(node.state):
                child_state, exec_info = executor.apply(node.state, repair)
                exec_info["answer_present"] = 1.0 if child_state.final_answer.strip() else 0.0
                exec_info["tactic_bias"] = action_bias_fn(node.state, repair) if action_bias_fn else 0.5
                exec_info["novelty_bonus"] = 1.0
                verifier_scores = _score_state(verifier, tokenizer, device, child_state)
                score = node.cumulative_score + combine_scores(
                    verifier_scores,
                    exec_info,
                    simplicity_penalty=float(score_config.get("simplicity_penalty", 0.02)),
                    invalid_penalty=float(score_config.get("invalid_penalty", 1.0)),
                    goal_bonus=float(score_config.get("goal_bonus", 0.4)),
                    solved_bonus=float(score_config.get("solved_bonus", 1.0)),
                    completion_bonus=float(score_config.get("completion_bonus", 0.15)),
                    incomplete_penalty=float(score_config.get("incomplete_penalty", 0.35)),
                    tactic_bonus=float(score_config.get("tactic_bonus", 0.12)),
                    value_weight=float(score_config.get("value_weight", 0.35)),
                    novelty_weight=float(score_config.get("novelty_weight", 0.08)),
                    depth=depth,
                    solved=(child_state.status == "solved"),
                )
                signature = state_signature_fn(child_state)
                accepted, novelty = transpositions.register(signature, score, depth)
                if not accepted:
                    continue
                if novelty != 1.0:
                    exec_info["novelty_bonus"] = novelty
                    score = node.cumulative_score + combine_scores(
                        verifier_scores,
                        exec_info,
                        simplicity_penalty=float(score_config.get("simplicity_penalty", 0.02)),
                        invalid_penalty=float(score_config.get("invalid_penalty", 1.0)),
                        goal_bonus=float(score_config.get("goal_bonus", 0.4)),
                        solved_bonus=float(score_config.get("solved_bonus", 1.0)),
                        completion_bonus=float(score_config.get("completion_bonus", 0.15)),
                        incomplete_penalty=float(score_config.get("incomplete_penalty", 0.35)),
                        tactic_bonus=float(score_config.get("tactic_bonus", 0.12)),
                        value_weight=float(score_config.get("value_weight", 0.35)),
                        novelty_weight=float(score_config.get("novelty_weight", 0.08)),
                        depth=depth,
                        solved=(child_state.status == "solved"),
                    )
                child = SearchNode(
                    state=child_state,
                    cumulative_score=score,
                    local_scores={**verifier_scores, **exec_info},
                    parent=node,
                    action=repair,
                    depth=depth,
                )
                candidates.append(child)
                explored.append(child)

        candidates.sort(key=lambda n: n.cumulative_score, reverse=True)
        beam = candidates[:beam_width] if candidates else beam
        if beam and beam[0].cumulative_score > best.cumulative_score:
            best = beam[0]
        if any(n.state.status == "solved" for n in beam):
            solved_nodes = [n for n in beam if n.state.status == "solved"]
            solved_nodes.sort(key=lambda n: n.cumulative_score, reverse=True)
            return solved_nodes[0].state, explored

    if event_logger is not None and best.state.status != "solved":
        event_logger(
            "search_budget_exhausted",
            domain=initial_state.domain,
            max_depth=max_depth,
            beam_width=beam_width,
            explored=len(explored),
        )
    return best.state, explored
