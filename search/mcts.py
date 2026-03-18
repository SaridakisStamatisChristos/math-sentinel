
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

from .beam import _build_exec_features, _build_prompt, _default_state_signature, _score_state, _should_prune_candidate
from .nodes import SearchNode
from .repair import fallback_repairs
from .scoring import combine_scores
from .transposition import TranspositionTable


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
    schema_provider: Any | None,
    transpositions: TranspositionTable,
    explored: List[SearchNode],
    event_logger: Optional[Callable[..., Any]],
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
        schema_provider=schema_provider,
    )

    candidates: List[MCTSNode] = []
    for txt in texts:
        actions, confidence = parse_actions_fn(txt)
        if not actions:
            if event_logger is not None:
                event_logger("invalid_schema_emission", domain=node.state.domain, depth=node.depth + 1, text=txt[:200])
            continue
        child_state = node.state
        exec_info: Dict[str, float] = {"goal_progress": 0.0, "valid_step": confidence}
        last_action = None
        last_action_state = node.state
        for action in actions:
            last_action_state = child_state
            child_state, local = executor.apply(child_state, action)
            exec_info["goal_progress"] = exec_info.get("goal_progress", 0.0) + float(local.get("goal_progress", 0.0))
            exec_info["valid_step"] = min(exec_info.get("valid_step", 1.0), float(local.get("valid_step", 1.0)))
            last_action = action
            if float(local.get("valid_step", 1.0)) <= 0.0 and event_logger is not None:
                event_logger(
                    "tool_failure",
                    domain=node.state.domain,
                    depth=node.depth + 1,
                    action=action.type.value,
                    tool=action.tool,
                    note=str(local.get("note", ""))[:200],
                )
            if child_state.status == "solved":
                break
        exec_info = _build_exec_features(child_state, exec_info, last_action, action_bias_fn, bias_state=last_action_state)
        if _should_prune_candidate(child_state, exec_info, score_config):
            if event_logger is not None:
                event_logger(
                    "branch_pruned",
                    domain=node.state.domain,
                    depth=node.depth + 1,
                    reason="invalid_or_stagnant",
                    action=getattr(last_action, "type", None).value if last_action is not None else "",
                    tool=getattr(last_action, "tool", "") if last_action is not None else "",
                )
            continue
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
            value_weight=float(score_config.get("value_weight", 0.35)),
            novelty_weight=float(score_config.get("novelty_weight", 0.08)),
            obligation_weight=float(score_config.get("obligation_weight", 0.08)),
            evidence_weight=float(score_config.get("evidence_weight", 0.10)),
            stagnation_penalty=float(score_config.get("stagnation_penalty", 0.18)),
            repeat_penalty=float(score_config.get("repeat_penalty", 0.10)),
            depth=node.depth + 1,
            solved=(child_state.status == "solved"),
        )
        total = node.cumulative_score + delta
        signature = state_signature_fn(child_state)
        accepted, novelty = transpositions.register(signature, total, node.depth + 1)
        if not accepted:
            continue
        if novelty != 1.0:
            exec_info["novelty_bonus"] = novelty
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
                value_weight=float(score_config.get("value_weight", 0.35)),
                novelty_weight=float(score_config.get("novelty_weight", 0.08)),
                obligation_weight=float(score_config.get("obligation_weight", 0.08)),
                evidence_weight=float(score_config.get("evidence_weight", 0.10)),
                stagnation_penalty=float(score_config.get("stagnation_penalty", 0.18)),
                repeat_penalty=float(score_config.get("repeat_penalty", 0.10)),
                depth=node.depth + 1,
                solved=(child_state.status == "solved"),
            )
            total = node.cumulative_score + delta
        if event_logger is not None and abs(float(verifier_scores.get("branch_priority", 0.0)) - float(verifier_scores.get("value_estimate", 0.0))) > 0.4:
            event_logger(
                "verifier_value_disagreement",
                domain=node.state.domain,
                depth=node.depth + 1,
                branch_priority=float(verifier_scores.get("branch_priority", 0.0)),
                value_estimate=float(verifier_scores.get("value_estimate", 0.0)),
            )
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

    repair_candidates = fallback_repairs_fn(node.state) if bool(score_config.get("enable_fallback_repairs", True)) else []
    for repair in repair_candidates:
        child_state, exec_info = executor.apply(node.state, repair)
        exec_info = _build_exec_features(child_state, exec_info, repair, action_bias_fn, bias_state=node.state)
        exec_info["fallback_repair_used"] = 1.0
        exec_info["tactic_bias"] = max(float(exec_info.get("tactic_bias", 0.5)), 1.0)
        exec_info["tool_bias"] = max(float(exec_info.get("tool_bias", 0.5)), 1.0)
        if event_logger is not None:
            event_logger("fallback_repair_used", domain=node.state.domain, depth=node.depth + 1, action=repair.type.value, tool=repair.tool)
        if _should_prune_candidate(child_state, exec_info, score_config):
            if event_logger is not None:
                event_logger(
                    "branch_pruned",
                    domain=node.state.domain,
                    depth=node.depth + 1,
                    reason="invalid_or_stagnant",
                    action=repair.type.value,
                    tool=repair.tool,
                )
            continue
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
            value_weight=float(score_config.get("value_weight", 0.35)),
            novelty_weight=float(score_config.get("novelty_weight", 0.08)),
            obligation_weight=float(score_config.get("obligation_weight", 0.08)),
            evidence_weight=float(score_config.get("evidence_weight", 0.10)),
            stagnation_penalty=float(score_config.get("stagnation_penalty", 0.18)),
            repeat_penalty=float(score_config.get("repeat_penalty", 0.10)),
            depth=node.depth + 1,
            solved=(child_state.status == "solved"),
        )
        delta += float(score_config.get("fallback_bonus", 0.0))
        total = node.cumulative_score + delta
        signature = state_signature_fn(child_state)
        accepted, novelty = transpositions.register(signature, total, node.depth + 1)
        if not accepted:
            continue
        if novelty != 1.0:
            exec_info["novelty_bonus"] = novelty
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
                value_weight=float(score_config.get("value_weight", 0.35)),
                novelty_weight=float(score_config.get("novelty_weight", 0.08)),
                obligation_weight=float(score_config.get("obligation_weight", 0.08)),
                evidence_weight=float(score_config.get("evidence_weight", 0.10)),
                stagnation_penalty=float(score_config.get("stagnation_penalty", 0.18)),
                repeat_penalty=float(score_config.get("repeat_penalty", 0.10)),
                depth=node.depth + 1,
                solved=(child_state.status == "solved"),
            )
            delta += float(score_config.get("fallback_bonus", 0.0))
            total = node.cumulative_score + delta
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
    schema_provider: Any | None = None,
    event_logger: Optional[Callable[..., Any]] = None,
) -> Tuple[ProofState, List[SearchNode]]:
    score_config = score_config or {}
    simulations = int(score_config.get("mcts_simulations", beam_width * max_depth * 2))
    exploration = float(score_config.get("mcts_exploration", 1.25))

    root = MCTSNode(state=initial_state.clone(), cumulative_score=0.0, depth=0, prior=1.0)
    explored: List[SearchNode] = [SearchNode(state=root.state, cumulative_score=0.0, depth=0)]
    best = root
    transpositions = TranspositionTable(capacity=int(score_config.get("transposition_capacity", 4096)))
    transpositions.register(state_signature_fn(root.state), 0.0, 0)

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
                schema_provider=schema_provider,
                transpositions=transpositions,
                explored=explored,
                event_logger=event_logger,
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
        root.children.sort(
            key=lambda child: (
                child.state.status == "solved",
                child.cumulative_score,
                child.q_value(),
                child.visits,
            ),
            reverse=True,
        )
        candidate = root.children[0]
        if candidate.state.status == "solved" or best.state.status != "solved":
            best = candidate if candidate.cumulative_score >= best.cumulative_score or candidate.state.status == "solved" else best
    if event_logger is not None and best.state.status != "solved":
        event_logger(
            "search_budget_exhausted",
            domain=initial_state.domain,
            max_depth=max_depth,
            simulations=simulations,
            explored=len(explored),
        )
    return best.state, explored
