#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import torch

from curriculum.generators import GeneratedTask
from curriculum.phases import PhaseScheduler
from domains import create_reasoning_domain, default_curriculum_config
from memory.hard_cases import HardCaseStore
from memory.lemma_store import LemmaStore
from memory.replay import ReplayBuffer
from memory.tactic_stats import TacticStats
from proof.executor import ProofExecutor
from proof.state import ProofState
from search.router import run_search
from sentinel.checkpointing import load_checkpoint, save_checkpoint
from sentinel.config import load_runtime_config, load_yaml
from sentinel.logging_utils import compact_metrics, log_jsonl, now_ts
from sentinel.losses import masked_ce, verifier_pairwise_loss
from sentinel.model_backends import build_model_runtime_info, create_prover_and_tokenizer
from sentinel.runtime import configure_runtime
from sentinel.runtime_events import build_runtime_event_logger
from sentinel.search_runtime import build_action_bias_fn, build_prompt_builder
from sentinel.verifier import StateVerifier


DEFAULT_DOMAIN = create_reasoning_domain("math")


def make_state(task: GeneratedTask) -> ProofState:
    return DEFAULT_DOMAIN.make_state(task)


def build_training_example(task: GeneratedTask) -> str:
    return DEFAULT_DOMAIN.build_training_example(task)


def build_verifier_examples(task: GeneratedTask) -> tuple[str, torch.Tensor, str, torch.Tensor]:
    return DEFAULT_DOMAIN.build_verifier_examples(task)


def build_verifier_targets(
    task: GeneratedTask,
    state: ProofState,
    local_scores: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    return DEFAULT_DOMAIN.build_verifier_targets(task, state, local_scores)


def pick_best_mined_pair(
    task: GeneratedTask,
    explored: List[Any],
    final_state: ProofState,
    reasoning_domain: Any = DEFAULT_DOMAIN,
) -> Optional[Dict[str, Any]]:
    positive_nodes = []
    negative_nodes = []
    root_node = explored[0] if explored else None

    for node in explored:
        state = node.state
        has_answer = bool(state.final_answer.strip())
        correct = has_answer and reasoning_domain.evaluate_answer(task, state.final_answer)
        if correct:
            positive_nodes.append(node)
        else:
            hardness = (
                float(node.local_scores.get("goal_progress", 0.0))
                + 0.35 * float(has_answer)
                + 0.15 * float(node.local_scores.get("valid_step", 0.0))
                - 0.05 * float(node.depth == 0)
            )
            negative_nodes.append((hardness, node))

    pos_node = max(positive_nodes, key=lambda n: n.cumulative_score, default=None)
    if pos_node is None and final_state.final_answer.strip() and reasoning_domain.evaluate_answer(task, final_state.final_answer):
        pos_state = final_state.clone()
        pos_target = reasoning_domain.build_verifier_targets(task, pos_state)
    elif pos_node is not None:
        pos_state = pos_node.state.clone()
        pos_target = reasoning_domain.build_verifier_targets(task, pos_state, pos_node.local_scores)
    else:
        return None

    if not negative_nodes and root_node is not None:
        negative_nodes.append((0.0, root_node))
    if not negative_nodes:
        return None

    negative_nodes.sort(key=lambda item: item[0], reverse=True)
    neg_node = negative_nodes[0][1]
    neg_state = neg_node.state.clone()
    neg_target = reasoning_domain.build_verifier_targets(task, neg_state, neg_node.local_scores)
    return {
        "kind": "verifier_pair",
        "domain": task.domain,
        "task": task.prompt,
        "pos_text": pos_state.serialize(),
        "pos_target": pos_target.tolist(),
        "neg_text": neg_state.serialize(),
        "neg_target": neg_target.tolist(),
    }


def sample_mined_verifier_pairs(replay: ReplayBuffer, limit: int) -> List[Dict[str, Any]]:
    pairs = [item for item in replay.sample_weighted(max(limit * 4, limit)) if item.get("kind") == "verifier_pair"]
    if len(pairs) < limit:
        extra = [item for item in replay.items if item.get("kind") == "verifier_pair"]
        if extra:
            pairs.extend(random.sample(extra, min(limit - len(pairs), len(extra))))
    return pairs[:limit]


def _normalize_verifier_target(values: Any) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float32)
    if tensor.numel() >= 6:
        return tensor[:6]
    if tensor.numel() == 5:
        valid_step, goal_progress, proof_completion, risk, branch_priority = [float(v) for v in tensor.tolist()]
        value_estimate = max(
            0.01,
            min(
                0.99,
                0.45 * goal_progress
                + 0.30 * proof_completion
                + 0.20 * branch_priority
                + 0.10 * valid_step
                - 0.15 * risk,
            ),
        )
        return torch.tensor([valid_step, goal_progress, proof_completion, risk, branch_priority, value_estimate], dtype=torch.float32)
    padded = torch.zeros(6, dtype=torch.float32)
    padded[: tensor.numel()] = tensor
    return padded


@torch.no_grad()
def mine_online_verifier_pairs(
    prover: torch.nn.Module,
    verifier: StateVerifier,
    tokenizer: Any,
    executor: ProofExecutor,
    phase: Any,
    replay: ReplayBuffer,
    device: str,
    count: int,
    *,
    beam_width: int,
    max_depth: int,
    proposal_count: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    score_config: Optional[Dict[str, Any]] = None,
    reasoning_domain: Any = DEFAULT_DOMAIN,
    prompt_builder: Optional[Any] = None,
    state_signature_fn: Optional[Any] = None,
    action_bias_fn: Optional[Any] = None,
    event_logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    mined: List[Dict[str, Any]] = []
    if count <= 0:
        return mined
    score_config = score_config or {}

    prover_was_training = prover.training
    verifier_was_training = verifier.training
    prover.eval()
    verifier.eval()
    try:
        for _ in range(count):
            task = reasoning_domain.sample_task(phase.domains)
            init = reasoning_domain.make_state(task)
            final_state, explored = run_search(
                prover=prover,
                verifier=verifier,
                tokenizer=tokenizer,
                executor=executor,
                initial_state=init,
                device=device,
                beam_width=beam_width,
                max_depth=max_depth,
                proposal_count=proposal_count,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                score_config=score_config,
                parse_actions_fn=reasoning_domain.parse_actions,
                fallback_repairs_fn=reasoning_domain.fallback_repairs,
                prompt_builder=prompt_builder or reasoning_domain.build_search_prompt,
                state_signature_fn=state_signature_fn or reasoning_domain.state_signature,
                action_bias_fn=action_bias_fn,
                schema_provider=reasoning_domain,
                event_logger=event_logger,
            )
            pair = pick_best_mined_pair(task, explored, final_state, reasoning_domain=reasoning_domain)
            if pair is not None:
                replay.add(pair)
                mined.append(pair)
    finally:
        prover.train(prover_was_training)
        verifier.train(verifier_was_training)
    return mined


def batch_encode(texts: List[str], tokenizer: Any, seq_len: int, device: str) -> torch.Tensor:
    ids = [tokenizer.encode(t, seq_len) for t in texts]
    return torch.tensor(ids, dtype=torch.long, device=device)


def load_persistent_memory(
    cfg: Dict[str, Any],
    replay: ReplayBuffer,
    lemma_store: LemmaStore,
    hard_cases: HardCaseStore,
    tactic_stats: TacticStats,
) -> None:
    replay.load_jsonl(cfg["memory"]["replay_path"])
    lemma_store.load(cfg["memory"]["lemma_store_path"])
    hard_cases.load(cfg["memory"]["hard_cases_path"])
    tactic_stats.load(cfg["memory"]["tactic_stats_path"])


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def verifier_init_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "hidden_size": int(cfg["verifier"]["hidden_size"]),
        "dropout": float(cfg["verifier"]["dropout"]),
        "n_heads": int(cfg["verifier"].get("n_heads", 4)),
        "n_layers": int(cfg["verifier"].get("n_layers", 2)),
        "max_seq_len": int(cfg["verifier"].get("max_seq_len", cfg["model"]["seq_len"])),
        "ff_mult": int(cfg["verifier"].get("ff_mult", 4)),
    }


def phase_online_mining_count(cfg: Dict[str, Any], phase_name: str) -> int:
    phase_counts = cfg["verifier"].get("online_pairs_by_phase", {})
    return int(phase_counts.get(phase_name, cfg["verifier"].get("online_pairs_per_step", 0)))


def phase_search_value(cfg: Dict[str, Any], phase_name: str, key: str, fallback: Any) -> Any:
    phase_values = cfg["verifier"].get(key, {})
    if isinstance(phase_values, dict):
        return phase_values.get(phase_name, fallback)
    return fallback


@torch.no_grad()
def run_eval(
    prover: torch.nn.Module,
    verifier: StateVerifier,
    tokenizer: Any,
    executor: ProofExecutor,
    scheduler: PhaseScheduler,
    step: int,
    device: str,
    eval_count: int = 24,
    beam_width: int = 4,
    max_depth: int = 4,
    proposal_count: int = 4,
    max_new_tokens: int = 72,
    temperature: float = 0.8,
    top_k: int = 24,
    score_config: Optional[Dict[str, Any]] = None,
    reasoning_domain: Any = DEFAULT_DOMAIN,
    prompt_builder: Optional[Any] = None,
    state_signature_fn: Optional[Any] = None,
    action_bias_fn: Optional[Any] = None,
    event_logger: Optional[Any] = None,
) -> Dict[str, float]:
    phase = scheduler.phase_for_step(step)
    solved = 0
    step_valid_total = 0.0
    branches_total = 0.0
    for _ in range(eval_count):
        task = reasoning_domain.sample_task(phase.domains)
        init = reasoning_domain.make_state(task)
        final_state, explored = run_search(
            prover=prover,
            verifier=verifier,
            tokenizer=tokenizer,
            executor=executor,
            initial_state=init,
            device=device,
            beam_width=beam_width,
            max_depth=max_depth,
            proposal_count=proposal_count,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            score_config=score_config,
            parse_actions_fn=reasoning_domain.parse_actions,
            fallback_repairs_fn=reasoning_domain.fallback_repairs,
            prompt_builder=prompt_builder or reasoning_domain.build_search_prompt,
            state_signature_fn=state_signature_fn or reasoning_domain.state_signature,
            action_bias_fn=action_bias_fn,
            schema_provider=reasoning_domain,
            event_logger=event_logger,
        )
        ok = final_state.status == "solved" and reasoning_domain.evaluate_answer(task, final_state.final_answer)
        solved += int(ok)
        branches_total += len(explored)
        if explored:
            step_valid_total += sum(n.local_scores.get("valid_step", 0.0) for n in explored[1:]) / max(1, len(explored) - 1)
    return {
        "eval_accuracy": solved / max(1, eval_count),
        "eval_step_validity": step_valid_total / max(1, eval_count),
        "eval_branches": branches_total / max(1, eval_count),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Math Sentinel V7")
    ap.add_argument("--backend", default="math")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--curriculum-config", default="")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--micro-batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model-provider", default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--adapter-mode", default=None)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--safe-runtime", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--checker-plugin", default="")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--memory-refresh-samples", type=int, default=None)
    args = ap.parse_args()

    cfg = load_runtime_config(args.config)
    curriculum_path = args.curriculum_config or default_curriculum_config(args.backend)
    curriculum_cfg = load_yaml(curriculum_path)
    scheduler = PhaseScheduler.from_dict(curriculum_cfg)

    if args.steps is not None:
        cfg["training"]["steps"] = args.steps
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.micro_batch_size is not None:
        cfg["training"]["micro_batch_size"] = args.micro_batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.device is not None:
        cfg["device"] = args.device
    if args.model_provider is not None:
        cfg["model"]["provider"] = args.model_provider
    if args.backbone is not None:
        cfg["model"]["backbone"] = args.backbone
    if args.adapter_mode is not None:
        cfg["model"]["adapter_mode"] = args.adapter_mode
    if args.local_files_only:
        cfg["model"]["local_files_only"] = True
    if args.eval_every is not None:
        cfg["training"]["eval_every"] = args.eval_every
    if args.save_every is not None:
        cfg["training"]["save_every"] = args.save_every
    if args.memory_refresh_samples is not None:
        cfg["training"]["memory_refresh_samples"] = args.memory_refresh_samples
    if args.compile:
        cfg["compile"] = True

    device = configure_runtime(cfg, deterministic_override=(True if args.deterministic else None), safe_override=(True if args.safe_runtime else None))
    Path(cfg["paths"]["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["logs_dir"]).mkdir(parents=True, exist_ok=True)
    event_logger = build_runtime_event_logger(cfg)

    prover, tokenizer = create_prover_and_tokenizer(cfg, device, for_training=True)
    training_prover: torch.nn.Module = prover
    verifier = StateVerifier(
        vocab_size=tokenizer.vocab_size,
        **verifier_init_kwargs(cfg),
    ).to(device)

    if cfg.get("compile") and hasattr(torch, "compile"):
        try:
            training_prover = cast(torch.nn.Module, torch.compile(prover))
        except Exception:
            pass

    reasoning_domain = create_reasoning_domain(args.backend, checker_plugin=args.checker_plugin)
    executor = reasoning_domain.create_executor()

    prover_optim = torch.optim.AdamW(prover.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    verifier_optim = torch.optim.AdamW(
        verifier.parameters(),
        lr=float(cfg["verifier"].get("lr", cfg["training"]["lr"])),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and bool(cfg["training"]["amp"])))

    replay = ReplayBuffer(capacity=int(cfg["memory"]["replay_capacity"]))
    lemma_store = LemmaStore()
    hard_cases = HardCaseStore(capacity=int(cfg["memory"]["hard_case_capacity"]))
    tactic_stats = TacticStats()
    load_persistent_memory(cfg, replay, lemma_store, hard_cases, tactic_stats)
    prompt_builder = build_prompt_builder(
        reasoning_domain,
        lemma_store=lemma_store,
        hard_case_store=hard_cases,
        tactic_stats=tactic_stats,
        retrieval_mode=str(cfg["memory"].get("retrieval_mode", "hybrid")),
        embedding_model=str(cfg["memory"].get("embedding_model", "hashing")),
        event_logger=event_logger,
    )
    action_bias_fn = build_action_bias_fn(tactic_stats)

    start_step = 1
    if args.resume:
        payload = load_checkpoint(args.resume, prover, verifier, prover_optim, verifier_optim, scaler=scaler, map_location=device)
        start_step = int(payload.get("step", 0)) + 1
        set_optimizer_lr(prover_optim, float(cfg["training"]["lr"]))
        set_optimizer_lr(verifier_optim, float(cfg["verifier"].get("lr", cfg["training"]["lr"])))

    log_path = str(Path(cfg["paths"]["logs_dir"]) / "train_v7.jsonl")
    print(f"[{now_ts()}] device={device} start_step={start_step} steps={cfg['training']['steps']}")

    for step in range(start_step, int(cfg["training"]["steps"]) + 1):
        phase = scheduler.phase_for_step(step)
        batch_size = int(cfg["training"]["batch_size"])
        micro_batch_size = max(1, int(cfg["training"].get("micro_batch_size", batch_size)))
        accum_steps = max(1, math.ceil(batch_size / micro_batch_size))

        examples: List[str] = []
        pos_texts: List[str] = []
        neg_texts: List[str] = []
        pos_targets: List[torch.Tensor] = []
        neg_targets: List[torch.Tensor] = []

        for _ in range(batch_size):
            task = reasoning_domain.sample_task(phase.domains)
            examples.append(reasoning_domain.build_training_example(task))
            pos_text, pos_t, neg_text, neg_t = reasoning_domain.build_verifier_examples(task)
            pos_texts.append(pos_text)
            neg_texts.append(neg_text)
            pos_targets.append(pos_t)
            neg_targets.append(neg_t)

        online_pairs: List[Dict[str, Any]] = []
        if step >= int(cfg["verifier"].get("online_mining_start_step", 1)):
            online_pair_count = phase_online_mining_count(cfg, phase.name)
            online_pairs = mine_online_verifier_pairs(
                prover=prover,
                verifier=verifier,
                tokenizer=tokenizer,
                executor=executor,
                phase=phase,
                replay=replay,
                device=device,
                count=online_pair_count,
                beam_width=int(phase_search_value(cfg, phase.name, "online_beam_width_by_phase", cfg["verifier"].get("online_beam_width", cfg["search"]["beam_width"]))),
                max_depth=int(phase_search_value(cfg, phase.name, "online_max_depth_by_phase", cfg["verifier"].get("online_max_depth", cfg["search"]["max_depth"]))),
                proposal_count=int(phase_search_value(cfg, phase.name, "online_proposal_count_by_phase", cfg["verifier"].get("online_proposal_count", cfg["search"]["proposal_count"]))),
                max_new_tokens=int(cfg["training"].get("max_new_tokens", 64)),
                temperature=float(phase_search_value(cfg, phase.name, "online_temperature_by_phase", cfg["verifier"].get("online_temperature", cfg["search"]["temperature"]))),
                top_k=int(phase_search_value(cfg, phase.name, "online_top_k_by_phase", cfg["verifier"].get("online_top_k", cfg["search"]["top_k"]))),
                score_config=cfg["search"],
                reasoning_domain=reasoning_domain,
                prompt_builder=prompt_builder,
                state_signature_fn=reasoning_domain.state_signature,
                action_bias_fn=action_bias_fn,
                event_logger=event_logger,
            )
            for pair in online_pairs:
                pos_texts.append(str(pair["pos_text"]))
                neg_texts.append(str(pair["neg_text"]))
                pos_targets.append(_normalize_verifier_target(pair["pos_target"]))
                neg_targets.append(_normalize_verifier_target(pair["neg_target"]))

        mined_pairs = sample_mined_verifier_pairs(replay, int(cfg["verifier"].get("mined_pairs_per_step", 0)))
        for pair in mined_pairs:
            pos_texts.append(str(pair["pos_text"]))
            neg_texts.append(str(pair["neg_text"]))
            pos_targets.append(_normalize_verifier_target(pair["pos_target"]))
            neg_targets.append(_normalize_verifier_target(pair["neg_target"]))

        training_prover.train()
        verifier.train()
        prover_optim.zero_grad(set_to_none=True)
        verifier_optim.zero_grad(set_to_none=True)

        verifier_micro_batch_size = max(1, math.ceil(len(pos_texts) / accum_steps))
        lm_loss_value = 0.0
        v_loss_value = 0.0
        v_bce_loss_value = 0.0
        v_rank_loss_value = 0.0
        verifier_loss_weight = float(cfg["verifier"].get("loss_weight", 0.4))
        amp_enabled = device == "cuda" and bool(cfg["training"]["amp"])

        for accum_idx in range(accum_steps):
            ex_start = accum_idx * micro_batch_size
            ex_end = min(ex_start + micro_batch_size, len(examples))
            pair_start = accum_idx * verifier_micro_batch_size
            pair_end = min(pair_start + verifier_micro_batch_size, len(pos_texts))
            if ex_start >= ex_end and pair_start >= pair_end:
                continue

            micro_total_loss: Optional[torch.Tensor] = None
            with torch.amp.autocast(device_type=("cuda" if device == "cuda" else "cpu"), enabled=amp_enabled):
                if ex_start < ex_end:
                    micro_examples = examples[ex_start:ex_end]
                    micro_batch = batch_encode(micro_examples, tokenizer, int(cfg["model"]["seq_len"]), device)
                    micro_x = micro_batch[:, :-1]
                    micro_y = micro_batch[:, 1:]
                    lm_loss = masked_ce(training_prover(micro_x), micro_y, tokenizer.pad_id)
                    lm_weight = len(micro_examples) / max(1, len(examples))
                    lm_loss_value += float(lm_loss.item()) * lm_weight
                    micro_total_loss = lm_loss * lm_weight

                if pair_start < pair_end:
                    micro_pos_texts = pos_texts[pair_start:pair_end]
                    micro_neg_texts = neg_texts[pair_start:pair_end]
                    pos_inputs = batch_encode(micro_pos_texts, tokenizer, int(cfg["model"]["seq_len"]), device)
                    neg_inputs = batch_encode(micro_neg_texts, tokenizer, int(cfg["model"]["seq_len"]), device)
                    pos_target_tensor = torch.stack(pos_targets[pair_start:pair_end], dim=0).to(device)
                    neg_target_tensor = torch.stack(neg_targets[pair_start:pair_end], dim=0).to(device)
                    pos_logits = verifier(pos_inputs)
                    neg_logits = verifier(neg_inputs)
                    v_loss, v_bce_loss, v_rank_loss = verifier_pairwise_loss(
                        pos_logits,
                        neg_logits,
                        pos_target_tensor,
                        neg_target_tensor,
                        margin=float(cfg["verifier"].get("margin", 0.2)),
                        rank_weight=float(cfg["verifier"].get("rank_weight", 0.35)),
                        focal_gamma=float(cfg["verifier"].get("focal_gamma", 2.0)),
                        focal_alpha=float(cfg["verifier"].get("focal_alpha", 0.75)),
                    )
                    v_weight = len(micro_pos_texts) / max(1, len(pos_texts))
                    v_loss_value += float(v_loss.item()) * v_weight
                    v_bce_loss_value += float(v_bce_loss.item()) * v_weight
                    v_rank_loss_value += float(v_rank_loss.item()) * v_weight
                    verifier_component = verifier_loss_weight * v_loss * v_weight
                    micro_total_loss = verifier_component if micro_total_loss is None else (micro_total_loss + verifier_component)

            if micro_total_loss is not None:
                scaler.scale(micro_total_loss).backward()

        total_loss_value = lm_loss_value + verifier_loss_weight * v_loss_value
        scaler.unscale_(prover_optim)
        scaler.unscale_(verifier_optim)
        torch.nn.utils.clip_grad_norm_(prover.parameters(), float(cfg["training"]["grad_clip"]))
        torch.nn.utils.clip_grad_norm_(verifier.parameters(), float(cfg["training"]["grad_clip"]))
        scaler.step(prover_optim)
        scaler.step(verifier_optim)
        scaler.update()

        metrics = {
            "step": step,
            "phase": phase.name,
            "lm_loss": lm_loss_value,
            "verifier_loss": v_loss_value,
            "verifier_bce_loss": v_bce_loss_value,
            "verifier_rank_loss": v_rank_loss_value,
            "verifier_online_pairs": float(len(online_pairs)),
            "total_loss": total_loss_value,
        }

        if step % int(cfg["training"]["log_every"]) == 0 or step == 1:
            print(f"[{now_ts()}] {compact_metrics(metrics)}")
            log_jsonl(log_path, metrics)

        if step % int(cfg["training"]["eval_every"]) == 0:
            prover.eval()
            verifier.eval()
            eval_metrics = run_eval(
                prover=prover,
                verifier=verifier,
                tokenizer=tokenizer,
                executor=executor,
                scheduler=scheduler,
                step=step,
                device=device,
                eval_count=16,
                beam_width=int(cfg["search"]["beam_width"]),
                max_depth=int(cfg["search"]["max_depth"]),
                proposal_count=int(cfg["search"]["proposal_count"]),
                max_new_tokens=int(cfg["training"]["max_new_tokens"]),
                temperature=float(cfg["search"]["temperature"]),
                top_k=int(cfg["search"]["top_k"]),
                score_config=cfg["search"],
                reasoning_domain=reasoning_domain,
                prompt_builder=prompt_builder,
                state_signature_fn=reasoning_domain.state_signature,
                action_bias_fn=action_bias_fn,
                event_logger=event_logger,
            )
            metrics.update(eval_metrics)
            print(f"[{now_ts()}] eval | {compact_metrics(eval_metrics)}")
            log_jsonl(log_path, metrics)

            # Refresh memory with a few sampled cases.
            for _ in range(int(cfg["training"].get("memory_refresh_samples", 4))):
                task = reasoning_domain.sample_task(phase.domains)
                init = reasoning_domain.make_state(task)
                final_state, explored = run_search(
                    prover=prover,
                    verifier=verifier,
                    tokenizer=tokenizer,
                    executor=executor,
                    initial_state=init,
                    device=device,
                    beam_width=int(cfg["search"]["beam_width"]),
                    max_depth=int(cfg["search"]["max_depth"]),
                    proposal_count=int(cfg["search"]["proposal_count"]),
                    max_new_tokens=int(cfg["training"]["max_new_tokens"]),
                    temperature=float(cfg["search"]["temperature"]),
                    top_k=int(cfg["search"]["top_k"]),
                    score_config=cfg["search"],
                    parse_actions_fn=reasoning_domain.parse_actions,
                    fallback_repairs_fn=reasoning_domain.fallback_repairs,
                    prompt_builder=prompt_builder,
                    state_signature_fn=reasoning_domain.state_signature,
                    action_bias_fn=action_bias_fn,
                    schema_provider=reasoning_domain,
                    event_logger=event_logger,
                )
                ok = final_state.status == "solved" and reasoning_domain.evaluate_answer(task, final_state.final_answer)
                replay.add({"task": task.prompt, "answer": final_state.final_answer or "<no_answer>", "ok": ok, "domain": task.domain})
                mined_pair = pick_best_mined_pair(task, explored, final_state, reasoning_domain=reasoning_domain)
                if mined_pair is not None:
                    mined_pair["weight"] = 1.5 if ok else 2.0
                    replay.add(mined_pair)
                if not ok:
                    hard_cases.add(
                        {
                            "task": task.prompt,
                            "domain": task.domain,
                            "score": len(explored),
                            "answer": final_state.final_answer or "<no_answer>",
                            "expected": task.answer,
                        }
                    )
                if ok:
                    lemma = reasoning_domain.maybe_derive_lemma(task)
                    if lemma is not None:
                        lemma_store.add(lemma)
                for node in explored:
                    if node.action is not None:
                        tactic_stats.record(task.domain, node.action.type.value, node.state.status == "solved")

        if step % int(cfg["training"]["save_every"]) == 0:
            ckpt_path = str(Path(cfg["paths"]["checkpoints_dir"]) / "last.pt")
            save_checkpoint(
                ckpt_path,
                prover=prover,
                verifier=verifier,
                prover_optim=prover_optim,
                verifier_optim=verifier_optim,
                scaler=scaler,
                step=step,
                config=cfg,
                extra_state={"phase": phase.name, "model_runtime": build_model_runtime_info(cfg, prover)},
            )
            replay.save_jsonl(cfg["memory"]["replay_path"])
            hard_cases.save(cfg["memory"]["hard_cases_path"])
            lemma_store.save(cfg["memory"]["lemma_store_path"])
            tactic_stats.save(cfg["memory"]["tactic_stats_path"])

    print(f"[{now_ts()}] training complete")


if __name__ == "__main__":
    main()
