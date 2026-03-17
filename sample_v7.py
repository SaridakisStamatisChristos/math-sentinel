#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any, Dict

import torch

from curriculum.phases import PhaseScheduler
from domains.math.backend import MathReasoningDomain
from search.beam import beam_search
from sentinel.checkpointing import load_checkpoint
from sentinel.config import load_runtime_config, load_yaml
from sentinel.model import TinyTransformerLM
from sentinel.tokenizer import build_default_tokenizer
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
    ap = argparse.ArgumentParser(description="Sample Math Sentinel V7")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--curriculum-config", default="config/curriculum.yaml")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--domain", default="")
    ap.add_argument("--problem", default="")
    ap.add_argument("--checker-plugin", default="")
    args = ap.parse_args()

    cfg = load_runtime_config(args.config)
    curriculum_cfg = load_yaml(args.curriculum_config)
    scheduler = PhaseScheduler.from_dict(curriculum_cfg)
    reasoning_domain = MathReasoningDomain(checker_plugin=args.checker_plugin)
    device = "cuda" if torch.cuda.is_available() and cfg["device"] in {"auto", "cuda"} else "cpu"

    tokenizer = build_default_tokenizer()
    prover = TinyTransformerLM(
        vocab_size=tokenizer.vocab_size,
        seq_len=int(cfg["model"]["seq_len"]),
        d_model=int(cfg["model"]["d_model"]),
        n_heads=int(cfg["model"]["n_heads"]),
        n_layers=int(cfg["model"]["n_layers"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    verifier = StateVerifier(
        vocab_size=tokenizer.vocab_size,
        **verifier_init_kwargs(cfg),
    ).to(device)

    if args.checkpoint:
        load_checkpoint(args.checkpoint, prover, verifier, map_location=device)

    executor = reasoning_domain.create_executor()

    if args.problem and args.domain:
        task = reasoning_domain.manual_task(args.domain, args.problem)
    else:
        phase = scheduler.phase_for_step(int(cfg["training"]["steps"]))
        task = reasoning_domain.sample_task(phase.domains)

    init = reasoning_domain.make_state(task)
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
        max_new_tokens=int(cfg["training"]["max_new_tokens"]),
        temperature=float(cfg["search"]["temperature"]),
        top_k=int(cfg["search"]["top_k"]),
        score_config=cfg["search"],
        parse_actions_fn=reasoning_domain.parse_actions,
        fallback_repairs_fn=reasoning_domain.fallback_repairs,
    )

    print("=== INPUT TASK ===")
    print(task.prompt)
    print("\n=== FINAL STATE ===")
    print(final_state.serialize())
    print("\n=== TRACE ===")
    print(reasoning_domain.render_human_trace(final_state))
    print("\n=== EXPLORED NODES ===")
    print(len(explored))


if __name__ == "__main__":
    main()
