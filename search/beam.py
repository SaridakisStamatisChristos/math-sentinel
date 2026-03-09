
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from proof.parser import parse_actions
from proof.state import ProofState
from proof.executor import ProofExecutor
from sentinel.generation import propose_actions
from sentinel.tokenizer import CharTokenizer
from sentinel.model import TinyTransformerLM
from sentinel.verifier import StateVerifier

from .nodes import SearchNode
from .repair import fallback_repairs
from .scoring import combine_scores


def _build_prompt(state: ProofState) -> str:
    instruction = (
        "\nEmit one or more actions using the form "
        '<action type="THINK">...</action> '
        '<action type="APPLY" tool="...">...</action> '
        '<action type="ANSWER">...</action>\n'
        "Always finish with an ANSWER action once you have enough information. "
        "Do not stop after THINK or APPLY if a final answer or proof sketch is available.\n"
    )
    return state.serialize() + instruction


def _score_state(verifier: StateVerifier, tokenizer: CharTokenizer, device: str, state: ProofState) -> Dict[str, float]:
    ids = tokenizer.encode(state.serialize(), seq_len=384)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        pred = verifier.predict_scores(x)
    return {k: float(v.item()) for k, v in pred.items()}


def beam_search(
    prover: TinyTransformerLM,
    verifier: StateVerifier,
    tokenizer: CharTokenizer,
    executor: ProofExecutor,
    initial_state: ProofState,
    device: str,
    beam_width: int = 4,
    max_depth: int = 4,
    proposal_count: int = 4,
    max_new_tokens: int = 72,
    temperature: float = 0.8,
    top_k: int = 24,
) -> Tuple[ProofState, List[SearchNode]]:
    root = SearchNode(state=initial_state.clone(), cumulative_score=0.0, depth=0)
    beam: List[SearchNode] = [root]
    explored: List[SearchNode] = [root]
    best = root

    for depth in range(1, max_depth + 1):
        candidates: List[SearchNode] = []
        for node in beam:
            if node.state.status == "solved":
                candidates.append(node)
                continue

            prompt = _build_prompt(node.state)
            texts = propose_actions(
                model=prover,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                proposal_count=proposal_count,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

            any_valid = False
            for txt in texts:
                actions, confidence = parse_actions(txt)
                if not actions:
                    continue
                any_valid = True
                state = node.state
                exec_info: Dict[str, float] = {"goal_progress": 0.0, "valid_step": confidence}
                child_state = state
                last_action = None
                for action in actions:
                    child_state, local = executor.apply(child_state, action)
                    exec_info["goal_progress"] = exec_info.get("goal_progress", 0.0) + float(local.get("goal_progress", 0.0))
                    exec_info["valid_step"] = min(exec_info.get("valid_step", 1.0), float(local.get("valid_step", 1.0)))
                    last_action = action
                    if child_state.status == "solved":
                        break
                exec_info["answer_present"] = 1.0 if child_state.final_answer.strip() else 0.0
                verifier_scores = _score_state(verifier, tokenizer, device, child_state)
                score = node.cumulative_score + combine_scores(
                    verifier_scores,
                    exec_info,
                    depth=depth,
                    solved=(child_state.status == "solved"),
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

            if not any_valid:
                for repair in fallback_repairs(node.state):
                    child_state, exec_info = executor.apply(node.state, repair)
                    exec_info["answer_present"] = 1.0 if child_state.final_answer.strip() else 0.0
                    verifier_scores = _score_state(verifier, tokenizer, device, child_state)
                    score = node.cumulative_score + combine_scores(
                        verifier_scores,
                        exec_info,
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

    return best.state, explored
