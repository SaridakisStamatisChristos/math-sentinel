
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from proof.parser import parse_actions
from proof.state import ProofState
from proof.executor import ProofExecutor
from sentinel.generation import propose_actions
from sentinel.model import TinyTransformerLM
from sentinel.tokenizer import StructuredTokenizer
from sentinel.verifier import StateVerifier

from .beam import _build_prompt, _default_state_signature, _score_state
from .nodes import SearchNode
from .repair import fallback_repairs
from .scoring import combine_scores


@dataclass
class MCTSNode:
    state: ProofState
    cumulative_score: float
    local_scores: Dict[str, float] = field(default_factory=dict)
    parent: Optional["MCTSNode"] = None
    action: Any = None
    depth: int = 0
    prior: float = 0.5
    visits: int = 0
    value_sum: float = 0.0
    children: List["MCTSNode"] = field(default_factory=list)
    expanded: bool = False

    def q_value(self) -> float:
        return self.value_sum / max(1, self.visits)


def _bounded_value(score: float) -> float:
    return 1.0 / (1.0 + math.exp(-score))


def _available_tools(executor: ProofExecutor) -> tuple[str, ...]:
    return tuple(getattr(getattr(executor, "tool_registry", None), "tools", {}).keys())


def _expand_node(
    node: MCTSNode,
    *,
    prover: TinyTransformerLM,
    verifier: StateVerifier,
    tokenizer: StructuredTokenizer,
    executor: ProofExecutor,
    device: str,
    proposal_count: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    score_config: Dict[str, Any],
    parse_actions_fn: Callable[[str], Tuple[List[Any], float]],
    fallback_repairs_fn: Callable[[ProofState], List[Any]],
    prompt_builder: Callable[[ProofState], str],
    state_signature_fn: Callable[[ProofState], str],
    action_bias_fn: Optional[Callable[[ProofState, Any], float]],
    seen_signatures: set[str],
    explored: List[SearchNode],
) -> List[MCTSNode]:
    if node.expanded or node.state.status == "solved":
        return node.children

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
        available_tools=_available_tools(executor),
        decoder_mode=str(score_config.get("decoder_mode", "hybrid")),
    )

    candidates: List[MCTSNode] = []
    for txt in texts:
        actions, confidence = parse_actions_fn(txt)
        if not actions:
            continue
        child_state = node.state
        exec_info: Dict[str, float] = {"goal_progress": 0.0, "valid_step": confidence}
        last_action = None
        for action in actions:
            child_state, local = executor.apply(child_state, action)
            exec_info["goal_progress"] = exec_info.get("goal_progress", 0.0) + float(local.get("goal_progress", 0.0))
            exec_info["valid_step"] = min(exec_info.get("valid_step", 1.0), float(local.get("valid_step", 1.0)))
            last_action = action
            if child_state.status == "solved":
                break
        exec_info["answer_present"] = 1.0 if child_state.final_answer.strip() else 0.0
        exec_info["tactic_bias"] = action_bias_fn(node.state, last_action) if action_bias_fn and last_action is not None else 0.5
        verifier_scores = _score_state(verifier, tokenizer, device, child_state)
        delta = combine_scores(
            verifier_scores,
            exec_info,
            simplicity_penalty=float(score_config.get("simplicity_penalty", 0.02)),
            invalid_penalty=float(score_config.get("invalid_penalty", 1.0)),
            goal_bonus=float(score_config.get("goal_bonus", 0.4)),
            solved_bonus=float(score_config.get("solved_bonus", 1.0)),
            completion_bonus=float(score_config.get("completion_bonus", 0.15)),
            incomplete_penalty=float(score_config.get("incomplete_penalty", 0.35)),
            tactic_bonus=float(score_config.get("tactic_bonus", 0.12)),
            depth=node.depth + 1,
            solved=(child_state.status == "solved"),
        )
        total = node.cumulative_score + delta
        signature = state_signature_fn(child_state)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        child = MCTSNode(
            state=child_state,
            cumulative_score=total,
            local_scores={**verifier_scores, **exec_info},
            parent=node,
            action=last_action,
            depth=node.depth + 1,
            prior=_bounded_value(delta),
        )
        candidates.append(child)
        explored.append(
            SearchNode(
                state=child_state,
                cumulative_score=total,
                local_scores=child.local_scores,
                parent=None,
                action=last_action,
                depth=child.depth,
            )
        )

    for repair in fallback_repairs_fn(node.state):
        child_state, exec_info = executor.apply(node.state, repair)
        exec_info["answer_present"] = 1.0 if child_state.final_answer.strip() else 0.0
        exec_info["tactic_bias"] = action_bias_fn(node.state, repair) if action_bias_fn else 0.5
        verifier_scores = _score_state(verifier, tokenizer, device, child_state)
        delta = combine_scores(
            verifier_scores,
            exec_info,
            simplicity_penalty=float(score_config.get("simplicity_penalty", 0.02)),
            invalid_penalty=float(score_config.get("invalid_penalty", 1.0)),
            goal_bonus=float(score_config.get("goal_bonus", 0.4)),
            solved_bonus=float(score_config.get("solved_bonus", 1.0)),
            completion_bonus=float(score_config.get("completion_bonus", 0.15)),
            incomplete_penalty=float(score_config.get("incomplete_penalty", 0.35)),
            tactic_bonus=float(score_config.get("tactic_bonus", 0.12)),
            depth=node.depth + 1,
            solved=(child_state.status == "solved"),
        )
        total = node.cumulative_score + delta
        signature = state_signature_fn(child_state)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        child = MCTSNode(
            state=child_state,
            cumulative_score=total,
            local_scores={**verifier_scores, **exec_info},
            parent=node,
            action=repair,
            depth=node.depth + 1,
            prior=_bounded_value(delta),
        )
        candidates.append(child)
        explored.append(
            SearchNode(
                state=child_state,
                cumulative_score=total,
                local_scores=child.local_scores,
                parent=None,
                action=repair,
                depth=child.depth,
            )
        )

    candidates.sort(key=lambda child: child.cumulative_score, reverse=True)
    node.children = candidates[: int(score_config.get("mcts_branching", max(2, proposal_count)))]
    node.expanded = True
    return node.children


def _select_child(node: MCTSNode, exploration: float) -> Optional[MCTSNode]:
    if not node.children:
        return None
    total_visits = max(1, node.visits)

    def _uct(child: MCTSNode) -> float:
        q = child.q_value()
        u = exploration * child.prior * math.sqrt(total_visits) / (1 + child.visits)
        return q + u

    return max(node.children, key=_uct)


def _backprop(node: MCTSNode, value: float) -> None:
    current: Optional[MCTSNode] = node
    while current is not None:
        current.visits += 1
        current.value_sum += value
        current = current.parent


def mcts_search(
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
) -> Tuple[ProofState, List[SearchNode]]:
    score_config = score_config or {}
    simulations = int(score_config.get("mcts_simulations", beam_width * max_depth * 2))
    exploration = float(score_config.get("mcts_exploration", 1.25))

    root = MCTSNode(state=initial_state.clone(), cumulative_score=0.0, depth=0, prior=1.0)
    explored: List[SearchNode] = [SearchNode(state=root.state, cumulative_score=0.0, depth=0)]
    best = root
    seen_signatures = {state_signature_fn(root.state)}

    for _ in range(simulations):
        node = root
        while node.expanded and node.children and node.depth < max_depth and node.state.status != "solved":
            selected = _select_child(node, exploration)
            if selected is None:
                break
            node = selected

        if node.depth < max_depth and node.state.status != "solved":
            children = _expand_node(
                node,
                prover=prover,
                verifier=verifier,
                tokenizer=tokenizer,
                executor=executor,
                device=device,
                proposal_count=proposal_count,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                score_config=score_config,
                parse_actions_fn=parse_actions_fn,
                fallback_repairs_fn=fallback_repairs_fn,
                prompt_builder=prompt_builder,
                state_signature_fn=state_signature_fn,
                action_bias_fn=action_bias_fn,
                seen_signatures=seen_signatures,
                explored=explored,
            )
            if children:
                node = children[0]

        value = _bounded_value(node.cumulative_score)
        if node.state.status == "solved":
            value = 1.0
        _backprop(node, value)
        if node.state.status == "solved" and node.cumulative_score >= best.cumulative_score:
            best = node
        elif best.state.status != "solved" and node.cumulative_score > best.cumulative_score:
            best = node

    if root.children:
        root.children.sort(key=lambda child: (child.state.status == "solved", child.visits, child.cumulative_score), reverse=True)
        candidate = root.children[0]
        if candidate.state.status == "solved" or best.state.status != "solved":
            best = candidate if candidate.cumulative_score >= best.cumulative_score or candidate.state.status == "solved" else best
    return best.state, explored
