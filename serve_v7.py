#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import torch

from domains import create_reasoning_domain
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


def solve_one(
    *,
    cfg: Dict[str, Any],
    reasoning_domain: Any,
    prover: torch.nn.Module,
    verifier: StateVerifier,
    tokenizer: Any,
    device: str,
    prompt_builder: Any,
    action_bias_fn: Any,
    event_logger: Any,
    domain: str,
    problem: str,
) -> Dict[str, Any]:
    task = reasoning_domain.manual_task(domain, problem)
    executor = reasoning_domain.create_executor()
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
    return {
        "domain": domain,
        "problem": problem,
        "status": final_state.status,
        "answer": final_state.final_answer,
        "explored_nodes": len(explored),
        "trace": reasoning_domain.render_human_trace(final_state),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve Math Sentinel V7 locally over stdin/stdout JSONL")
    ap.add_argument("--backend", default="math")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--model-provider", default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--safe-runtime", action="store_true")
    ap.add_argument("--checker-plugin", default="")
    ap.add_argument("--stdin-jsonl", action="store_true")
    ap.add_argument("--domain", default="")
    ap.add_argument("--problem", default="")
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
    reasoning_domain = create_reasoning_domain(args.backend, checker_plugin=args.checker_plugin, runtime_config=cfg)
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
    verifier = StateVerifier(vocab_size=tokenizer.vocab_size, **verifier_init_kwargs(cfg)).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, prover, verifier, map_location=device)
    prover.eval()
    verifier.eval()

    if args.stdin_jsonl:
        while True:
            try:
                raw = input()
            except EOFError:
                break
            line = raw.strip()
            if not line:
                continue
            request = json.loads(line)
            print(
                json.dumps(
                    solve_one(
                        cfg=cfg,
                        reasoning_domain=reasoning_domain,
                        prover=prover,
                        verifier=verifier,
                        tokenizer=tokenizer,
                        device=device,
                        prompt_builder=prompt_builder,
                        action_bias_fn=action_bias_fn,
                        event_logger=event_logger,
                        domain=str(request.get("domain", "")),
                        problem=str(request.get("problem", "")),
                    ),
                    ensure_ascii=False,
                )
            )
        return

    if not args.domain or not args.problem:
        raise SystemExit("--domain and --problem are required unless --stdin-jsonl is used")
    print(
        json.dumps(
            solve_one(
                cfg=cfg,
                reasoning_domain=reasoning_domain,
                prover=prover,
                verifier=verifier,
                tokenizer=tokenizer,
                device=device,
                prompt_builder=prompt_builder,
                action_bias_fn=action_bias_fn,
                event_logger=event_logger,
                domain=args.domain,
                problem=args.problem,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
