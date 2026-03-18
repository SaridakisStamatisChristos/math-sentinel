#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any, Dict, List

import torch

from domains import available_backends, create_reasoning_domain
from search.router import run_search
from sentinel.checkpointing import load_checkpoint
from sentinel.config import load_runtime_config
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


def resolve_backends(spec: str) -> List[str]:
    if not spec or spec == "all":
        return available_backends()
    names = [name.strip() for name in spec.split(",") if name.strip()]
    return names or available_backends()


def run_backend_benchmark(
    backend_name: str,
    cfg: Dict[str, Any],
    prover: torch.nn.Module,
    verifier: StateVerifier,
    tokenizer: Any,
    device: str,
    checker_plugin: str,
    event_logger: Any,
) -> Dict[str, Any]:
    reasoning_domain = create_reasoning_domain(backend_name, checker_plugin=checker_plugin)
    executor = reasoning_domain.create_executor()
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
    action_bias_fn = build_action_bias_fn(tactic_stats)

    solved = 0
    equivalent = 0
    branches = 0
    tasks = reasoning_domain.benchmark_tasks()
    for task in tasks:
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
        solved += int(final_state.status == "solved")
        equivalent += int(reasoning_domain.evaluate_answer(task, final_state.final_answer) if final_state.final_answer else False)
        branches += len(explored)

    return {
        "backend": backend_name,
        "tasks": len(tasks),
        "solved_rate": solved / max(1, len(tasks)),
        "equivalence_rate": equivalent / max(1, len(tasks)),
        "avg_branches": branches / max(1, len(tasks)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark Math Sentinel V7 backends")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--model-provider", default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--safe-runtime", action="store_true")
    ap.add_argument("--backends", default="all")
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
    event_logger = build_runtime_event_logger(cfg)
    prover, tokenizer = create_prover_and_tokenizer(cfg, device, for_training=False)
    verifier = StateVerifier(vocab_size=tokenizer.vocab_size, **verifier_init_kwargs(cfg)).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, prover, verifier, map_location=device)
    prover.eval()
    verifier.eval()

    for backend_name in resolve_backends(args.backends):
        print(run_backend_benchmark(backend_name, cfg, prover, verifier, tokenizer, device, args.checker_plugin, event_logger))


if __name__ == "__main__":
    main()
