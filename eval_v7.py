#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any, Dict

import torch

from curriculum.phases import PhaseScheduler
from domains import create_reasoning_domain, default_curriculum_config
from search.router import run_search
from sentinel.checkpointing import load_checkpoint
from sentinel.config import load_runtime_config, load_yaml
from sentinel.model_backends import create_prover_and_tokenizer
from sentinel.runtime import configure_runtime
from sentinel.runtime_events import build_runtime_event_logger
from sentinel.search_runtime import build_action_bias_fn, build_prompt_builder, load_search_memory
from sentinel.verifier import StateVerifier


def verifier_init_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "hidden_size": int(cfg["verifier"]["hidden_size"]),
        "dropout": float(cfg["verifier"]["dropout"]),
        "n_heads": int(cfg["verifier"].get("n_heads", 4)),
        "n_layers": int(cfg["verifier"].get("n_layers", 2)),
        "max_seq_len": int(cfg["verifier"].get("max_seq_len", cfg["model"]["seq_len"])),
        "ff_mult": int(cfg["verifier"].get("ff_mult", 4)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Math Sentinel V7")
    ap.add_argument("--backend", default="math")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--curriculum-config", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--model-provider", default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--safe-runtime", action="store_true")
    ap.add_argument("--count", type=int, default=48)
    ap.add_argument("--step", type=int, default=2000)
    ap.add_argument("--checker-plugin", default="")
    args = ap.parse_args()

    cfg = load_runtime_config(args.config)
    if args.model_provider is not None:
        cfg["model"]["provider"] = args.model_provider
    if args.backbone is not None:
        cfg["model"]["backbone"] = args.backbone
    if args.local_files_only:
        cfg["model"]["local_files_only"] = True
    device = configure_runtime(cfg, deterministic_override=(True if args.deterministic else None), safe_override=(True if args.safe_runtime else None))
    curriculum_path = args.curriculum_config or default_curriculum_config(args.backend)
    curriculum_cfg = load_yaml(curriculum_path)
    scheduler = PhaseScheduler.from_dict(curriculum_cfg)
    phase = scheduler.phase_for_step(args.step)
    reasoning_domain = create_reasoning_domain(args.backend, checker_plugin=args.checker_plugin, runtime_config=cfg)
    event_logger = build_runtime_event_logger(cfg)
    lemma_store, hard_cases, tactic_stats = load_search_memory(cfg)
    prompt_builder = build_prompt_builder(
        reasoning_domain,
        lemma_store=lemma_store,
        hard_case_store=hard_cases,
        tactic_stats=tactic_stats,
        retrieval_mode=str(cfg["memory"].get("retrieval_mode", "hybrid")),
        embedding_model=str(cfg["memory"].get("embedding_model", "hashing")),
        event_logger=event_logger,
    )
    action_bias_fn = build_action_bias_fn(tactic_stats, reasoning_domain=reasoning_domain)

    prover, tokenizer = create_prover_and_tokenizer(cfg, device, for_training=False)
    verifier = StateVerifier(
        vocab_size=tokenizer.vocab_size,
        **verifier_init_kwargs(cfg),
    ).to(device)

    executor = reasoning_domain.create_executor()

    if args.checkpoint:
        load_checkpoint(args.checkpoint, prover, verifier, map_location=device)

    prover.eval()
    verifier.eval()

    solved = 0
    equivalence_solved = 0
    branches = 0
    for _ in range(args.count):
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
        ok = final_state.status == "solved"
        eq_ok = reasoning_domain.evaluate_answer(task, final_state.final_answer) if final_state.final_answer else False
        solved += int(ok)
        equivalence_solved += int(eq_ok)
        branches += len(explored)

    print({
        "phase": phase.name,
        "count": args.count,
        "solved_rate": solved / max(1, args.count),
        "equivalence_rate": equivalence_solved / max(1, args.count),
        "avg_branches": branches / max(1, args.count),
    })


if __name__ == "__main__":
    main()
