from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import torch

from benchmarks.integrity import collect_state_audit, ensure_benchmark_audit, is_runtime_oracle_field
from benchmarks.official_corpus import ensure_official_manifest, resolve_official_corpus_selection
from domains import available_backends, create_reasoning_domain
from engine.task import ReasoningTask
from search.router import run_search
from sentinel.checkpointing import load_checkpoint
from sentinel.model_backends import create_prover_and_tokenizer
from sentinel.search_runtime import build_action_bias_fn, build_prompt_builder, load_search_memory
from sentinel.verifier import StateVerifier

from .base import BenchmarkCaseResult, BenchmarkSuite, BenchmarkSuiteResult
from .manifest_loader import load_manifest_suite
from .public_catalog import available_public_suites, available_public_suite_groups, load_public_suite


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


def resolve_suite_targets(suite_spec: str, backends_spec: str) -> List[Tuple[str, str]]:
    targets: List[Tuple[str, str]] = []
    raw_suite = (suite_spec or "internal").strip()
    normalized_suite = raw_suite.lower()
    public_suites = set(available_public_suites())
    public_groups = available_public_suite_groups()

    if normalized_suite.startswith("manifest:"):
        manifest_path = raw_suite.split(":", 1)[1].strip()
        if manifest_path:
            return [("manifest", manifest_path)]
        return []

    if normalized_suite.startswith("official:"):
        selection = raw_suite.split(":", 1)[1].strip() or "all"
        return [("official", name) for name in resolve_official_corpus_selection(selection)]

    if normalized_suite in {"internal", "all"}:
        for backend_name in resolve_backends(backends_spec):
            targets.append(("internal", backend_name))

    if normalized_suite in {"all", "public_all"}:
        for suite_name in public_groups["public_all"]:
            targets.append(("public", suite_name))
    elif normalized_suite in public_groups:
        for suite_name in public_groups[normalized_suite]:
            targets.append(("public", suite_name))
    elif normalized_suite in public_suites:
        targets.append(("public", normalized_suite))

    return targets


def _case_audit_from_search(cfg: Dict[str, Any], task: ReasoningTask, final_state: Any, explored: List[Any]) -> Dict[str, Any]:
    assistance_mode = str(cfg.get("benchmark", {}).get("assistance_mode", "unassisted")).lower()
    audit = ensure_benchmark_audit({}, assistance_mode=assistance_mode)
    touched_fields: List[str] = []
    integrity_events: List[str] = []
    runtime_oracle_fields: List[str] = []
    guided_rollout_used = False
    fallback_repair_used = False
    fallback_repair_attempts = 0
    guided_rollout_steps = 0
    fallback_chain_used = False
    fallback_chain_steps = 0
    environment_issues: List[str] = []

    states = [getattr(node, "state", None) for node in explored] + [final_state]
    for state in states:
        metadata = getattr(state, "metadata", {}) if state is not None else {}
        if not isinstance(metadata, dict):
            continue
        state_audit = collect_state_audit(metadata)
        for field in state_audit.get("oracle_fields_touched", []):
            text = str(field).strip()
            if text and text not in touched_fields:
                touched_fields.append(text)
        for event in state_audit.get("integrity_events", []):
            text = str(event).strip()
            if text and text not in integrity_events:
                integrity_events.append(text)
        for key in metadata.keys():
            key_text = str(key)
            if is_runtime_oracle_field(key_text) and key_text not in runtime_oracle_fields:
                runtime_oracle_fields.append(key_text)
        search_audit = metadata.get("search_audit", {})
        if isinstance(search_audit, dict):
            guided_rollout_used = guided_rollout_used or bool(search_audit.get("guided_rollout_used", False))
            fallback_repair_used = fallback_repair_used or bool(search_audit.get("fallback_repair_used", False))
            fallback_repair_attempts = max(fallback_repair_attempts, int(search_audit.get("fallback_repair_attempts", 0)))
            guided_rollout_steps = max(guided_rollout_steps, int(search_audit.get("guided_rollout_steps", 0)))
            fallback_chain_used = fallback_chain_used or bool(search_audit.get("fallback_chain_used", False))
            fallback_chain_steps = max(fallback_chain_steps, int(search_audit.get("fallback_chain_steps", 0)))
        failure_summary = metadata.get("last_test_failure_summary", {})
        if isinstance(failure_summary, dict):
            for issue in failure_summary.get("environment_issues", []):
                text = str(issue).strip()
                if text and text not in environment_issues:
                    environment_issues.append(text)

    for node in explored:
        local_scores = getattr(node, "local_scores", {}) or {}
        if float(local_scores.get("guided_rollout_used", 0.0)) > 0.0:
            guided_rollout_used = True
            guided_rollout_steps += 1
        if float(local_scores.get("fallback_repair_used", 0.0)) > 0.0:
            fallback_repair_used = True
            fallback_repair_attempts += 1
        if float(local_scores.get("fallback_chain_used", 0.0)) > 0.0:
            fallback_chain_used = True
            fallback_chain_steps += 1

    audit.update(
        {
            "guided_rollout_used": guided_rollout_used,
            "guided_rollout_steps": guided_rollout_steps,
            "fallback_repair_used": fallback_repair_used,
            "fallback_repair_attempts": fallback_repair_attempts,
            "fallback_chain_used": fallback_chain_used,
            "fallback_chain_steps": fallback_chain_steps,
            "oracle_fields_touched": touched_fields,
            "oracle_fields_present_in_runtime": bool(runtime_oracle_fields),
            "integrity_events": integrity_events,
            "environment_issues": environment_issues,
        }
    )
    integrity_passed = not runtime_oracle_fields and not touched_fields
    if assistance_mode != "unassisted":
        integrity_passed = True
    audit["benchmark_integrity_passed"] = integrity_passed
    audit["task_id"] = str(getattr(task, "task_id", ""))
    return audit


def _claim_profile_requirements(cfg: Dict[str, Any], audit: Dict[str, Any]) -> tuple[bool, List[str]]:
    if not bool(cfg.get("benchmark", {}).get("claim_mode", False)):
        return True, []
    failures: List[str] = []
    if not bool(audit.get("benchmark_integrity_passed", False)):
        failures.append("benchmark_integrity_passed=false")
    if bool(audit.get("fallback_chain_used", False)):
        failures.append("fallback_chain_used=true")
    if bool(audit.get("fallback_repair_used", False)):
        failures.append("fallback_repair_used=true")
    if bool(audit.get("guided_rollout_used", False)):
        failures.append("guided_rollout_used=true")
    return (not failures), failures


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
    reasoning_domain = create_reasoning_domain(backend_name, checker_plugin=checker_plugin, runtime_config=cfg)
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
    action_bias_fn = build_action_bias_fn(tactic_stats, reasoning_domain=reasoning_domain)

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
        case_audit = _case_audit_from_search(cfg, task, final_state, explored)
        claim_ok, claim_failures = _claim_profile_requirements(cfg, case_audit)
        case_audit["claim_profile_passed"] = claim_ok
        if not bool(case_audit.get("benchmark_integrity_passed", True)):
            if event_logger is not None:
                event_logger(
                    "benchmark_integrity_violation",
                    suite=suite_name,
                    backend=backend_name,
                    task_id=str(getattr(task, "task_id", "")),
                    oracle_fields_touched="|".join(str(item) for item in case_audit.get("oracle_fields_touched", [])),
                    runtime_oracle_fields=str(case_audit.get("oracle_fields_present_in_runtime", False)),
                )
            if bool(cfg.get("benchmark", {}).get("fail_on_integrity_violation", True)):
                raise RuntimeError(
                    f"benchmark integrity violation for suite={suite_name} task={getattr(task, 'task_id', '')}: "
                    f"{case_audit.get('integrity_events', [])}"
                )
        if not claim_ok:
            if event_logger is not None:
                event_logger(
                    "benchmark_claim_path_violation",
                    suite=suite_name,
                    backend=backend_name,
                    task_id=str(getattr(task, "task_id", "")),
                    failures="|".join(claim_failures),
                )
            if bool(cfg.get("benchmark", {}).get("fail_on_integrity_violation", True)):
                raise RuntimeError(
                    f"benchmark claim-path violation for suite={suite_name} task={getattr(task, 'task_id', '')}: "
                    f"{claim_failures}"
                )
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
                audit=case_audit,
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
        metadata={
            "report_lane": str(cfg.get("benchmark", {}).get("report_lane", "")).strip(),
            "task_count": len(task_list),
            "benchmark_integrity_passed": all(bool(case.audit.get("benchmark_integrity_passed", True)) for case in case_results),
            "claim_profile_passed": all(bool(case.audit.get("claim_profile_passed", True)) for case in case_results),
            "guided_rollout_used": any(bool(case.audit.get("guided_rollout_used", False)) for case in case_results),
            "fallback_repair_used": any(bool(case.audit.get("fallback_repair_used", False)) for case in case_results),
            "fallback_chain_used": any(bool(case.audit.get("fallback_chain_used", False)) for case in case_results),
            "integrity_pass_rate": sum(1 for case in case_results if bool(case.audit.get("benchmark_integrity_passed", False))) / max(1, len(case_results)),
            "claim_pass_rate": sum(1 for case in case_results if bool(case.audit.get("claim_profile_passed", False))) / max(1, len(case_results)),
            "guided_rollout_rate": sum(1 for case in case_results if bool(case.audit.get("guided_rollout_used", False))) / max(1, len(case_results)),
            "fallback_repair_rate": sum(1 for case in case_results if bool(case.audit.get("fallback_repair_used", False))) / max(1, len(case_results)),
            "fallback_chain_rate": sum(1 for case in case_results if bool(case.audit.get("fallback_chain_used", False))) / max(1, len(case_results)),
            "environment_issue_counts": {
                issue: sum(1 for case in case_results if issue in case.audit.get("environment_issues", []))
                for issue in sorted(
                    {
                        str(issue)
                        for case in case_results
                        for issue in case.audit.get("environment_issues", [])
                        if str(issue).strip()
                    }
                )
            },
            "oracle_fields_touched": sorted(
                {
                    str(field)
                    for case in case_results
                    for field in case.audit.get("oracle_fields_touched", [])
                    if str(field).strip()
                }
            ),
        },
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
    reasoning_domain = create_reasoning_domain(backend_name, checker_plugin=checker_plugin, runtime_config=cfg)
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


def run_suite_target(
    target_kind: str,
    target_name: str,
    cfg: Dict[str, Any],
    prover: torch.nn.Module,
    verifier: StateVerifier,
    tokenizer: Any,
    device: str,
    checker_plugin: str,
    event_logger: Any,
    max_cases: int | None = None,
) -> BenchmarkSuiteResult:
    if target_kind == "internal":
        return run_backend_benchmark(target_name, cfg, prover, verifier, tokenizer, device, checker_plugin, event_logger)
    if target_kind == "public":
        return run_public_suite(target_name, cfg, prover, verifier, tokenizer, device, checker_plugin, event_logger)
    if target_kind == "manifest":
        return run_manifest_suite(target_name, cfg, prover, verifier, tokenizer, device, checker_plugin, event_logger, max_cases=max_cases)
    if target_kind == "official":
        manifest_path = ensure_official_manifest(target_name, cfg, strict_materialization=True)
        return run_manifest_suite(manifest_path, cfg, prover, verifier, tokenizer, device, checker_plugin, event_logger, max_cases=max_cases)
    raise ValueError(f"unknown benchmark target kind: {target_kind}")


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


def run_manifest_suite(
    manifest_path: str,
    cfg: Dict[str, Any],
    prover: torch.nn.Module,
    verifier: StateVerifier,
    tokenizer: Any,
    device: str,
    checker_plugin: str,
    event_logger: Any,
    max_cases: int | None = None,
) -> BenchmarkSuiteResult:
    suite: BenchmarkSuite = load_manifest_suite(manifest_path, max_cases=max_cases)
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
