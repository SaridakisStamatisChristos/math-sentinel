#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
import torch
import yaml

from curriculum.generators import GeneratedTask, sample_task
from curriculum.oracle import evaluate_answer
from curriculum.phases import PhaseScheduler
from curriculum.trajectory_builder import build_gold_trace
from memory.hard_cases import HardCaseStore
from memory.lemma_store import LemmaStore
from memory.replay import ReplayBuffer
from memory.tactic_stats import TacticStats
from proof.lemmas import (
    derive_linear_lemma,
    derive_polynomial_lemma,
    derive_calculus_lemma,
    derive_logic_lemma,
    derive_arithmetic_lemma,
    derive_fractions_lemma,
    derive_divmod_lemma,
    derive_gcd_lcm_lemma,
    derive_modular_lemma,
    derive_primality_lemma,
    derive_factorization_lemma,
    derive_parity_proof_lemma,
)
from proof.executor import ProofExecutor
from proof.state import ProofState
from search.beam import beam_search
from sentinel.checkpointing import load_checkpoint, save_checkpoint
from sentinel.logging_utils import compact_metrics, log_jsonl, now_ts
from sentinel.losses import masked_ce, verifier_pairwise_loss
from sentinel.model import TinyTransformerLM
from sentinel.tokenizer import build_default_tokenizer
from sentinel.verifier import StateVerifier
from tools.registry import ToolRegistry


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def device_from_cfg(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def make_state(task: GeneratedTask) -> ProofState:
    return ProofState(
        task_id=task.task_id,
        domain=task.domain,
        problem_text=task.prompt,
        goal=task.goal,
        expected_answer=task.answer,
        metadata=task.meta,
    )


def build_training_example(task: GeneratedTask) -> str:
    state = make_state(task)
    return state.serialize() + "\n" + build_gold_trace(task)


def build_verifier_examples(task: GeneratedTask) -> tuple[str, torch.Tensor, str, torch.Tensor]:
    pos = make_state(task)
    pos.final_answer = task.answer
    pos.status = "solved"
    pos.derived_facts.append(task.answer)
    pos.action_history.append({"type": "ANSWER", "content": task.answer})
    pos.tool_history.append({"tool": "oracle", "result": {"ok": True, "answer": task.answer}})

    neg = make_state(task)
    neg.derived_facts.append("search_not_started")

    pos_t = build_verifier_targets(task, pos)
    neg_t = build_verifier_targets(task, neg, local_scores={"valid_step": 0.35, "goal_progress": 0.0, "risk_score": 0.8})
    return pos.serialize(), pos_t, neg.serialize(), neg_t


def build_verifier_targets(
    task: GeneratedTask,
    state: ProofState,
    local_scores: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    local_scores = local_scores or {}
    has_answer = bool(state.final_answer.strip())
    correct = has_answer and evaluate_answer(task, state.final_answer)
    solved = state.status == "solved"

    valid_step = float(local_scores.get("valid_step", 1.0 if solved or has_answer else 0.6))
    explicit_progress = float(local_scores.get("goal_progress", 0.0))
    structural_progress = min(
        1.0,
        0.12 * len(state.derived_facts)
        + 0.08 * len(state.tool_history)
        + 0.06 * len(state.action_history)
        + 0.08 * len(state.lemma_refs)
        + 0.04 * len(state.subgoals),
    )
    goal_progress = max(explicit_progress, structural_progress)
    if correct:
        goal_progress = max(goal_progress, 0.95)
    elif has_answer:
        goal_progress = min(0.75, max(goal_progress, 0.3))

    proof_completion = 1.0 if correct and solved else (0.25 if has_answer else min(0.2, goal_progress * 0.5))
    risk = float(local_scores.get("risk_score", 0.05 if correct else (0.85 if has_answer else 0.45)))
    branch_priority = max(0.05, min(0.98, 0.55 * goal_progress + 0.25 * valid_step + 0.20 * proof_completion))

    values = [
        max(0.02, min(0.99, valid_step)),
        max(0.0, min(0.99, goal_progress)),
        max(0.0, min(0.99, proof_completion)),
        max(0.01, min(0.99, risk)),
        branch_priority,
    ]
    return torch.tensor(values, dtype=torch.float32)


def pick_best_mined_pair(task: GeneratedTask, explored: List[Any], final_state: ProofState) -> Optional[Dict[str, Any]]:
    positive_nodes = []
    negative_nodes = []
    root_node = explored[0] if explored else None

    for node in explored:
        state = node.state
        has_answer = bool(state.final_answer.strip())
        correct = has_answer and evaluate_answer(task, state.final_answer)
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
    if pos_node is None and final_state.final_answer.strip() and evaluate_answer(task, final_state.final_answer):
        pos_state = final_state.clone()
        pos_target = build_verifier_targets(task, pos_state)
    elif pos_node is not None:
        pos_state = pos_node.state.clone()
        pos_target = build_verifier_targets(task, pos_state, pos_node.local_scores)
    else:
        return None

    if not negative_nodes and root_node is not None:
        negative_nodes.append((0.0, root_node))
    if not negative_nodes:
        return None

    negative_nodes.sort(key=lambda item: item[0], reverse=True)
    neg_node = negative_nodes[0][1]
    neg_state = neg_node.state.clone()
    neg_target = build_verifier_targets(task, neg_state, neg_node.local_scores)
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
    pairs = [item for item in replay.sample(max(limit * 4, limit)) if item.get("kind") == "verifier_pair"]
    if len(pairs) < limit:
        extra = [item for item in replay.items if item.get("kind") == "verifier_pair"]
        if extra:
            pairs.extend(random.sample(extra, min(limit - len(pairs), len(extra))))
    return pairs[:limit]


@torch.no_grad()
def mine_online_verifier_pairs(
    prover: TinyTransformerLM,
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
) -> List[Dict[str, Any]]:
    mined: List[Dict[str, Any]] = []
    if count <= 0:
        return mined

    prover_was_training = prover.training
    verifier_was_training = verifier.training
    prover.eval()
    verifier.eval()
    try:
        for _ in range(count):
            task = sample_task(phase.domains)
            init = make_state(task)
            final_state, explored = beam_search(
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
            )
            pair = pick_best_mined_pair(task, explored, final_state)
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


@torch.no_grad()
def run_eval(
    prover: TinyTransformerLM,
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
) -> Dict[str, float]:
    phase = scheduler.phase_for_step(step)
    solved = 0
    step_valid_total = 0.0
    branches_total = 0.0
    for _ in range(eval_count):
        task = sample_task(phase.domains)
        init = make_state(task)
        final_state, explored = beam_search(
            prover=prover,
            verifier=verifier,
            tokenizer=tokenizer,
            executor=executor,
            initial_state=init,
            device=device,
            beam_width=beam_width,
            max_depth=max_depth,
            proposal_count=proposal_count,
        )
        ok = final_state.status == "solved" and evaluate_answer(task, final_state.final_answer)
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
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--curriculum-config", default="config/curriculum.yaml")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--micro-batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--checker-plugin", default="")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--memory-refresh-samples", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    curriculum_cfg = load_yaml(args.curriculum_config)
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
    if args.eval_every is not None:
        cfg["training"]["eval_every"] = args.eval_every
    if args.save_every is not None:
        cfg["training"]["save_every"] = args.save_every
    if args.memory_refresh_samples is not None:
        cfg["training"]["memory_refresh_samples"] = args.memory_refresh_samples
    if args.compile:
        cfg["compile"] = True

    set_seed(int(cfg["seed"]))
    device = device_from_cfg(cfg["device"])
    Path(cfg["paths"]["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["logs_dir"]).mkdir(parents=True, exist_ok=True)

    tokenizer = build_default_tokenizer()
    prover = TinyTransformerLM(
        vocab_size=tokenizer.vocab_size,
        seq_len=int(cfg["model"]["seq_len"]),
        d_model=int(cfg["model"]["d_model"]),
        n_heads=int(cfg["model"]["n_heads"]),
        n_layers=int(cfg["model"]["n_layers"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    training_prover: torch.nn.Module = prover
    verifier = StateVerifier(
        vocab_size=tokenizer.vocab_size,
        hidden_size=int(cfg["verifier"]["hidden_size"]),
        dropout=float(cfg["verifier"]["dropout"]),
        n_heads=int(cfg["verifier"].get("n_heads", 4)),
        n_layers=int(cfg["verifier"].get("n_layers", 2)),
        max_seq_len=int(cfg["verifier"].get("max_seq_len", cfg["model"]["seq_len"])),
        ff_mult=int(cfg["verifier"].get("ff_mult", 4)),
    ).to(device)

    if cfg.get("compile") and hasattr(torch, "compile"):
        try:
            training_prover = cast(torch.nn.Module, torch.compile(prover))
        except Exception:
            pass

    registry = ToolRegistry()
    if args.checker_plugin:
        registry.load_plugin(args.checker_plugin)
    executor = ProofExecutor(registry)

    prover_optim = torch.optim.AdamW(prover.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    verifier_optim = torch.optim.AdamW(
        verifier.parameters(),
        lr=float(cfg["verifier"].get("lr", cfg["training"]["lr"])),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and bool(cfg["training"]["amp"])))

    replay = ReplayBuffer(capacity=int(cfg["memory"]["replay_capacity"]))
    lemma_store = LemmaStore()
    hard_cases = HardCaseStore(capacity=int(cfg["memory"]["hard_case_capacity"]))
    tactic_stats = TacticStats()
    load_persistent_memory(cfg, replay, lemma_store, hard_cases, tactic_stats)

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

        examples: List[str] = []
        pos_texts: List[str] = []
        neg_texts: List[str] = []
        pos_targets: List[torch.Tensor] = []
        neg_targets: List[torch.Tensor] = []

        for _ in range(int(cfg["training"]["batch_size"])):
            task = sample_task(phase.domains)
            examples.append(build_training_example(task))
            pos_text, pos_t, neg_text, neg_t = build_verifier_examples(task)
            pos_texts.append(pos_text)
            neg_texts.append(neg_text)
            pos_targets.append(pos_t)
            neg_targets.append(neg_t)

        online_pairs: List[Dict[str, Any]] = []
        if step >= int(cfg["verifier"].get("online_mining_start_step", 1)):
            online_pairs = mine_online_verifier_pairs(
                prover=prover,
                verifier=verifier,
                tokenizer=tokenizer,
                executor=executor,
                phase=phase,
                replay=replay,
                device=device,
                count=int(cfg["verifier"].get("online_pairs_per_step", 0)),
                beam_width=int(cfg["verifier"].get("online_beam_width", cfg["search"]["beam_width"])),
                max_depth=int(cfg["verifier"].get("online_max_depth", cfg["search"]["max_depth"])),
                proposal_count=int(cfg["verifier"].get("online_proposal_count", cfg["search"]["proposal_count"])),
                max_new_tokens=int(cfg["training"].get("max_new_tokens", 64)),
                temperature=float(cfg["verifier"].get("online_temperature", cfg["search"]["temperature"])),
                top_k=int(cfg["verifier"].get("online_top_k", cfg["search"]["top_k"])),
            )
            for pair in online_pairs:
                pos_texts.append(str(pair["pos_text"]))
                neg_texts.append(str(pair["neg_text"]))
                pos_targets.append(torch.tensor(pair["pos_target"], dtype=torch.float32))
                neg_targets.append(torch.tensor(pair["neg_target"], dtype=torch.float32))

        mined_pairs = sample_mined_verifier_pairs(replay, int(cfg["verifier"].get("mined_pairs_per_step", 0)))
        for pair in mined_pairs:
            pos_texts.append(str(pair["pos_text"]))
            neg_texts.append(str(pair["neg_text"]))
            pos_targets.append(torch.tensor(pair["pos_target"], dtype=torch.float32))
            neg_targets.append(torch.tensor(pair["neg_target"], dtype=torch.float32))

        batch = batch_encode(examples, tokenizer, int(cfg["model"]["seq_len"]), device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        training_prover.train()
        verifier.train()
        prover_optim.zero_grad(set_to_none=True)
        verifier_optim.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=("cuda" if device == "cuda" else "cpu"), enabled=(device == "cuda" and bool(cfg["training"]["amp"]))):
            logits = training_prover(x)
            lm_loss = masked_ce(logits, y, tokenizer.pad_id)

            pos_inputs = batch_encode(pos_texts, tokenizer, int(cfg["model"]["seq_len"]), device)
            neg_inputs = batch_encode(neg_texts, tokenizer, int(cfg["model"]["seq_len"]), device)
            pos_target_tensor = torch.stack(pos_targets, dim=0).to(device)
            neg_target_tensor = torch.stack(neg_targets, dim=0).to(device)
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

            total_loss = lm_loss + float(cfg["verifier"].get("loss_weight", 0.4)) * v_loss

        scaler.scale(total_loss).backward()
        scaler.unscale_(prover_optim)
        torch.nn.utils.clip_grad_norm_(prover.parameters(), float(cfg["training"]["grad_clip"]))
        torch.nn.utils.clip_grad_norm_(verifier.parameters(), float(cfg["training"]["grad_clip"]))
        scaler.step(prover_optim)
        scaler.step(verifier_optim)
        scaler.update()

        metrics = {
            "step": step,
            "phase": phase.name,
            "lm_loss": float(lm_loss.item()),
            "verifier_loss": float(v_loss.item()),
            "verifier_bce_loss": float(v_bce_loss.item()),
            "verifier_rank_loss": float(v_rank_loss.item()),
            "verifier_online_pairs": float(len(online_pairs)),
            "total_loss": float(total_loss.item()),
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
            )
            metrics.update(eval_metrics)
            print(f"[{now_ts()}] eval | {compact_metrics(eval_metrics)}")
            log_jsonl(log_path, metrics)

            # Refresh memory with a few sampled cases.
            for _ in range(int(cfg["training"].get("memory_refresh_samples", 4))):
                task = sample_task(phase.domains)
                init = make_state(task)
                final_state, explored = beam_search(
                    prover=prover,
                    verifier=verifier,
                    tokenizer=tokenizer,
                    executor=executor,
                    initial_state=init,
                    device=device,
                    beam_width=int(cfg["search"]["beam_width"]),
                    max_depth=int(cfg["search"]["max_depth"]),
                    proposal_count=int(cfg["search"]["proposal_count"]),
                )
                ok = final_state.status == "solved" and evaluate_answer(task, final_state.final_answer)
                replay.add({"task": task.prompt, "answer": final_state.final_answer or "<no_answer>", "ok": ok, "domain": task.domain})
                mined_pair = pick_best_mined_pair(task, explored, final_state)
                if mined_pair is not None:
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
                    if task.domain == "linear_equation":
                        lemma_store.add(derive_linear_lemma(task.prompt))
                    elif task.domain == "polynomial_simplify":
                        lemma_store.add(derive_polynomial_lemma(task.prompt))
                    elif task.domain in ["derivative", "integral"]:
                        lemma_store.add(derive_calculus_lemma(task.prompt))
                    elif task.domain == "logic":
                        lemma_store.add(derive_logic_lemma(task.prompt))
                    elif task.domain == "arithmetic":
                        lemma_store.add(derive_arithmetic_lemma(task.prompt))
                    elif task.domain == "fractions":
                        lemma_store.add(derive_fractions_lemma(task.prompt))
                    elif task.domain == "divmod":
                        lemma_store.add(derive_divmod_lemma(task.prompt))
                    elif task.domain == "gcd_lcm":
                        lemma_store.add(derive_gcd_lcm_lemma(task.prompt))
                    elif task.domain == "modular":
                        lemma_store.add(derive_modular_lemma(task.prompt))
                    elif task.domain == "primality":
                        lemma_store.add(derive_primality_lemma(task.prompt))
                    elif task.domain == "factorization":
                        lemma_store.add(derive_factorization_lemma(task.prompt))
                    elif task.domain == "parity_proof":
                        lemma_store.add(derive_parity_proof_lemma(task.prompt))
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
                extra_state={"phase": phase.name},
            )
            replay.save_jsonl(cfg["memory"]["replay_path"])
            hard_cases.save(cfg["memory"]["hard_cases_path"])
            lemma_store.save(cfg["memory"]["lemma_store_path"])
            tactic_stats.save(cfg["memory"]["tactic_stats_path"])

    print(f"[{now_ts()}] training complete")


if __name__ == "__main__":
    main()
