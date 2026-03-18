from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch

from domains import available_backends, create_reasoning_domain
from engine.task import ReasoningTask
from search.router import run_search
from sentinel.checkpointing import load_checkpoint
from sentinel.model_backends import create_prover_and_tokenizer
from sentinel.search_runtime import build_action_bias_fn, build_prompt_builder, load_search_memory
from sentinel.verifier import StateVerifier

from .base import BenchmarkCaseResult, BenchmarkSuite, BenchmarkSuiteResult
from .public_catalog import load_public_suite


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


def load_benchmark_runtime(
    cfg: Dict[str, Any],
    device: str,
    checkpoint: str = "",
) -> tuple[torch.nn.Module, Any, StateVerifier]:
    prover, tokenizer = create_prover_and_tokenizer(cfg, device, for_training=False)
    verifier = StateVerifier(vocab_size=tokenizer.vocab_size, **verifier_init_kwargs(cfg)).to(device)
    if checkpoint:
        load_checkpoint(checkpoint, prover, verifier, map_location=device)
    prover.eval()
    verifier.eval()
    return prover, tokenizer, verifier


def run_task_collection(
    suite_name: str,
    tier: str,
    description: str,
    backend_name: str,
    tasks: Iterable[ReasoningTask],
    cfg: Dict[str, Any],
    prover: torch.nn.Module,
    verifier: StateVerifier,
    tokenizer: Any,
    device: str,
    checker_plugin: str,
    event_logger: Any,
) -> BenchmarkSuiteResult:
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
    case_results: List[BenchmarkCaseResult] = []
    task_list = list(tasks)
    for task in task_list:
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
        case_solved = final_state.status == "solved"
        case_equivalent = reasoning_domain.evaluate_answer(task, final_state.final_answer) if final_state.final_answer else False
        solved += int(case_solved)
        equivalent += int(case_equivalent)
        branches += len(explored)
        case_results.append(
            BenchmarkCaseResult(
                task_id=str(getattr(task, "task_id", "")),
                domain=str(getattr(task, "domain", backend_name)),
                prompt=str(getattr(task, "prompt", "")),
                expected_answer=str(getattr(task, "answer", "")),
                final_answer=str(getattr(final_state, "final_answer", "")),
                status=str(getattr(final_state, "status", "")),
                solved=case_solved,
                equivalent=case_equivalent,
                explored_nodes=len(explored),
                metadata=dict(getattr(task, "meta", {})),
            )
        )

    return BenchmarkSuiteResult(
        suite=suite_name,
        backend=backend_name,
        tier=tier,
        description=description,
        solved_rate=solved / max(1, len(task_list)),
        equivalence_rate=equivalent / max(1, len(task_list)),
        avg_branches=branches / max(1, len(task_list)),
        cases=case_results,
        metadata={"task_count": len(task_list)},
    )


def run_backend_benchmark(
    backend_name: str,
    cfg: Dict[str, Any],
    prover: torch.nn.Module,
    verifier: StateVerifier,
    tokenizer: Any,
    device: str,
    checker_plugin: str,
    event_logger: Any,
) -> BenchmarkSuiteResult:
    reasoning_domain = create_reasoning_domain(backend_name, checker_plugin=checker_plugin)
    return run_task_collection(
        suite_name=f"internal_{backend_name}",
        tier="internal",
        description=f"Built-in benchmark suite for backend {backend_name}.",
        backend_name=backend_name,
        tasks=reasoning_domain.benchmark_tasks(),
        cfg=cfg,
        prover=prover,
        verifier=verifier,
        tokenizer=tokenizer,
        device=device,
        checker_plugin=checker_plugin,
        event_logger=event_logger,
    )


def run_public_suite(
    suite_name: str,
    cfg: Dict[str, Any],
    prover: torch.nn.Module,
    verifier: StateVerifier,
    tokenizer: Any,
    device: str,
    checker_plugin: str,
    event_logger: Any,
) -> BenchmarkSuiteResult:
    suite: BenchmarkSuite = load_public_suite(suite_name)
    return run_task_collection(
        suite_name=suite.name,
        tier=suite.tier,
        description=suite.description,
        backend_name=suite.backend,
        tasks=suite.cases,
        cfg=cfg,
        prover=prover,
        verifier=verifier,
        tokenizer=tokenizer,
        device=device,
        checker_plugin=checker_plugin,
        event_logger=event_logger,
    )
