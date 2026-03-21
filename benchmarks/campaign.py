from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, List, Tuple

from sentinel.model_backends import build_model_runtime_info
from sentinel.config import deep_merge_dicts
from sentinel.runtime import configure_runtime
from sentinel.runtime_events import build_runtime_event_logger

from .ablations import BenchmarkAblation, apply_benchmark_ablation
from .base import BenchmarkSuiteResult
from .profiles import BenchmarkProfile, apply_benchmark_profile
from .results import append_jsonl, save_json, save_suite_result, save_text
from .runners import load_benchmark_runtime, resolve_suite_targets, run_suite_target


def _slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "run"


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / max(1, len(items))


def _git_value(*args: str) -> str:
    try:
        output = subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL)
        return output.strip()
    except Exception:
        return ""


def collect_git_context() -> Dict[str, str]:
    return {
        "branch": _git_value("branch", "--show-current"),
        "commit": _git_value("rev-parse", "--short", "HEAD"),
        "commit_full": _git_value("rev-parse", "HEAD"),
    }


def _aggregate_suite_results(results: List[BenchmarkSuiteResult]) -> Tuple[float, float, float]:
    return (
        _mean([result.solved_rate for result in results]),
        _mean([result.equivalence_rate for result in results]),
        _mean([result.avg_branches for result in results]),
    )


def _aggregate_audit_metrics(results: List[BenchmarkSuiteResult]) -> Dict[str, float]:
    cases = [case for result in results for case in result.cases]
    if not cases:
        metadata = [result.metadata for result in results]
        return {
            "integrity_pass_rate": _mean([1.0 if bool(item.get("benchmark_integrity_passed", False)) else 0.0 for item in metadata]),
            "claim_pass_rate": _mean([1.0 if bool(item.get("claim_profile_passed", False)) else 0.0 for item in metadata]),
            "guided_rollout_rate": _mean([1.0 if bool(item.get("guided_rollout_used", False)) else 0.0 for item in metadata]),
            "fallback_repair_rate": _mean([1.0 if bool(item.get("fallback_repair_used", False)) else 0.0 for item in metadata]),
            "fallback_chain_rate": _mean([1.0 if bool(item.get("fallback_chain_used", False)) else 0.0 for item in metadata]),
            "oracle_touch_rate": _mean([1.0 if bool(item.get("oracle_fields_touched")) else 0.0 for item in metadata]),
        }
    return {
        "integrity_pass_rate": _mean([1.0 if bool(case.audit.get("benchmark_integrity_passed", False)) else 0.0 for case in cases]),
        "claim_pass_rate": _mean([1.0 if bool(case.audit.get("claim_profile_passed", False)) else 0.0 for case in cases]),
        "guided_rollout_rate": _mean([1.0 if bool(case.audit.get("guided_rollout_used", False)) else 0.0 for case in cases]),
        "fallback_repair_rate": _mean([1.0 if bool(case.audit.get("fallback_repair_used", False)) else 0.0 for case in cases]),
        "fallback_chain_rate": _mean([1.0 if bool(case.audit.get("fallback_chain_used", False)) else 0.0 for case in cases]),
        "oracle_touch_rate": _mean([1.0 if bool(case.audit.get("oracle_fields_touched", [])) else 0.0 for case in cases]),
    }


@dataclass
class BenchmarkCampaignRun:
    profile: str
    ablation: str
    repeat_index: int
    seed: int
    suite_results: List[BenchmarkSuiteResult]
    model_runtime: Dict[str, Any]
    git: Dict[str, str]
    config: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def aggregate_metrics(self) -> Dict[str, float]:
        solved_rate, equivalence_rate, avg_branches = _aggregate_suite_results(self.suite_results)
        aggregate = {
            "solved_rate": solved_rate,
            "equivalence_rate": equivalence_rate,
            "avg_branches": avg_branches,
        }
        aggregate.update(_aggregate_audit_metrics(self.suite_results))
        return aggregate

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["suite_results"] = [result.to_dict() for result in self.suite_results]
        payload["aggregate"] = self.aggregate_metrics()
        return payload


@dataclass
class BenchmarkCampaignSummary:
    name: str
    suite_spec: str
    targets: List[str]
    repeat: int
    runs: List[BenchmarkCampaignRun]
    variants: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["runs"] = [run.to_dict() for run in self.runs]
        return payload


def summarize_campaign_runs(runs: List[BenchmarkCampaignRun]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[BenchmarkCampaignRun]] = {}
    for run in runs:
        grouped.setdefault((run.profile, run.ablation), []).append(run)

    variants: List[Dict[str, Any]] = []
    for (profile, ablation), group in grouped.items():
        suite_groups: Dict[str, List[BenchmarkSuiteResult]] = {}
        for run in group:
            for result in run.suite_results:
                suite_groups.setdefault(result.suite, []).append(result)
        aggregate_metrics = [run.aggregate_metrics() for run in group]
        aggregate_solved = _mean([item["solved_rate"] for item in aggregate_metrics])
        aggregate_equivalence = _mean([item["equivalence_rate"] for item in aggregate_metrics])
        aggregate_branches = _mean([item["avg_branches"] for item in aggregate_metrics])
        aggregate_integrity = _mean([item["integrity_pass_rate"] for item in aggregate_metrics])
        aggregate_claim = _mean([item["claim_pass_rate"] for item in aggregate_metrics])
        aggregate_guided = _mean([item["guided_rollout_rate"] for item in aggregate_metrics])
        aggregate_repairs = _mean([item["fallback_repair_rate"] for item in aggregate_metrics])
        aggregate_chain = _mean([item["fallback_chain_rate"] for item in aggregate_metrics])
        aggregate_oracle = _mean([item["oracle_touch_rate"] for item in aggregate_metrics])
        stable_core = len(
            {
                (
                    round(item["solved_rate"], 6),
                    round(item["equivalence_rate"], 6),
                    round(item["integrity_pass_rate"], 6),
                    round(item["claim_pass_rate"], 6),
                    round(item["guided_rollout_rate"], 6),
                    round(item["fallback_repair_rate"], 6),
                    round(item["fallback_chain_rate"], 6),
                )
                for item in aggregate_metrics
            }
        ) == 1
        branch_values = [item["avg_branches"] for item in aggregate_metrics]
        branch_variance = max(branch_values) - min(branch_values) if branch_values else 0.0
        stable = stable_core and branch_variance <= 0.5
        suite_breakdown = []
        for suite_name, suite_runs in sorted(suite_groups.items()):
            audit = _aggregate_audit_metrics(suite_runs)
            suite_breakdown.append(
                {
                    "suite": suite_name,
                    "solved_rate": _mean([entry.solved_rate for entry in suite_runs]),
                    "equivalence_rate": _mean([entry.equivalence_rate for entry in suite_runs]),
                    "avg_branches": _mean([entry.avg_branches for entry in suite_runs]),
                    "integrity_pass_rate": audit["integrity_pass_rate"],
                    "claim_pass_rate": audit["claim_pass_rate"],
                    "guided_rollout_rate": audit["guided_rollout_rate"],
                    "fallback_repair_rate": audit["fallback_repair_rate"],
                    "fallback_chain_rate": audit["fallback_chain_rate"],
                }
            )
        lane = str(group[0].metadata.get("report_lane", group[0].metadata.get("profile_metadata", {}).get("purpose", "unspecified"))).strip() or "unspecified"
        variants.append(
            {
                "profile": profile,
                "ablation": ablation,
                "lane": lane,
                "repeats": len(group),
                "stable": stable,
                "stable_core": stable_core,
                "branch_variance": branch_variance,
                "aggregate": {
                    "solved_rate": aggregate_solved,
                    "equivalence_rate": aggregate_equivalence,
                    "avg_branches": aggregate_branches,
                    "integrity_pass_rate": aggregate_integrity,
                    "claim_pass_rate": aggregate_claim,
                    "guided_rollout_rate": aggregate_guided,
                    "fallback_repair_rate": aggregate_repairs,
                    "fallback_chain_rate": aggregate_chain,
                    "oracle_touch_rate": aggregate_oracle,
                },
                "suite_breakdown": suite_breakdown,
            }
        )

    baseline_lookup = {
        variant["profile"]: variant["aggregate"]
        for variant in variants
        if variant["ablation"] == "baseline"
    }
    for variant in variants:
        baseline = baseline_lookup.get(variant["profile"])
        if baseline is None:
            variant["delta_vs_baseline"] = {
                "solved_rate": 0.0,
                "equivalence_rate": 0.0,
                "avg_branches": 0.0,
            }
            continue
        variant["delta_vs_baseline"] = {
            "solved_rate": variant["aggregate"]["solved_rate"] - baseline["solved_rate"],
            "equivalence_rate": variant["aggregate"]["equivalence_rate"] - baseline["equivalence_rate"],
            "avg_branches": variant["aggregate"]["avg_branches"] - baseline["avg_branches"],
        }
    variants.sort(key=lambda item: (item["profile"], item["ablation"]))
    return variants


def render_campaign_report(summary: BenchmarkCampaignSummary) -> str:
    lines = [
        f"# Benchmark Campaign: {summary.name}",
        "",
        f"- suite spec: `{summary.suite_spec}`",
        f"- targets: `{', '.join(summary.targets)}`",
        f"- repeats: `{summary.repeat}`",
        f"- branch: `{summary.metadata.get('git', {}).get('branch', '')}`",
        f"- commit: `{summary.metadata.get('git', {}).get('commit', '')}`",
        f"- runtime fingerprints: `{len(summary.metadata.get('runtime_fingerprints', []))}`",
        "",
        "## Variants",
        "",
    ]
    lane_groups: Dict[str, List[Dict[str, Any]]] = {}
    for variant in summary.variants:
        lane_groups.setdefault(str(variant.get("lane", "unspecified")), []).append(variant)
    for lane, lane_variants in sorted(lane_groups.items()):
        lines.append(f"### Lane: {lane}")
        for variant in lane_variants:
            aggregate = variant["aggregate"]
            delta = variant["delta_vs_baseline"]
            lines.append(
                (
                    f"- `{variant['profile']}/{variant['ablation']}` "
                    f"solved={aggregate['solved_rate']:.3f} "
                    f"equiv={aggregate['equivalence_rate']:.3f} "
                    f"branches={aggregate['avg_branches']:.3f} "
                    f"integrity={aggregate['integrity_pass_rate']:.3f} "
                    f"claim={aggregate['claim_pass_rate']:.3f} "
                    f"repairs={aggregate['fallback_repair_rate']:.3f} "
                    f"guided={aggregate['guided_rollout_rate']:.3f} "
                    f"stable={variant['stable']} "
                    f"delta_equiv={delta['equivalence_rate']:+.3f}"
                )
            )
        lines.append("")
    lines.append("## Publication Summary")
    lines.append("")
    for variant in summary.variants:
        aggregate = variant["aggregate"]
        lines.append(
            (
                f"- `{variant['profile']}/{variant['ablation']}` "
                f"lane={variant.get('lane', 'unspecified')} "
                f"integrity_pass_rate={aggregate['integrity_pass_rate']:.3f} "
                f"claim_pass_rate={aggregate['claim_pass_rate']:.3f} "
                f"fallback_repair_rate={aggregate['fallback_repair_rate']:.3f} "
                f"fallback_chain_rate={aggregate['fallback_chain_rate']:.3f} "
                f"guided_rollout_rate={aggregate['guided_rollout_rate']:.3f} "
                f"oracle_touch_rate={aggregate['oracle_touch_rate']:.3f} "
                f"branch_variance={variant.get('branch_variance', 0.0):.3f}"
            )
        )
    lines.append("")
    lines.append("## Suite Breakdown")
    lines.append("")
    for variant in summary.variants:
        lines.append(f"### {variant['profile']} / {variant['ablation']}")
        for suite in variant["suite_breakdown"]:
            lines.append(
                (
                    f"- `{suite['suite']}` solved={suite['solved_rate']:.3f} "
                    f"equiv={suite['equivalence_rate']:.3f} "
                    f"branches={suite['avg_branches']:.3f} "
                    f"integrity={suite['integrity_pass_rate']:.3f} "
                    f"claim={suite['claim_pass_rate']:.3f} "
                    f"repairs={suite['fallback_repair_rate']:.3f} "
                    f"guided={suite['guided_rollout_rate']:.3f}"
                )
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def run_benchmark_campaign(
    *,
    base_cfg: Dict[str, Any],
    cfg_overrides: Dict[str, Any] | None,
    suite_spec: str,
    backends_spec: str,
    profiles: List[BenchmarkProfile],
    ablations: List[BenchmarkAblation],
    checkpoint: str,
    results_dir: str,
    checker_plugin: str,
    deterministic_override: bool | None,
    safe_override: bool | None,
    repeat: int,
    campaign_name: str,
    max_cases: int | None = None,
) -> BenchmarkCampaignSummary:
    targets = resolve_suite_targets(suite_spec, backends_spec)
    if not targets:
        raise ValueError(f"no benchmark targets resolved for suite={suite_spec!r} backends={backends_spec!r}")

    git_context = collect_git_context()
    campaign = _slug(campaign_name or f"{suite_spec}_campaign")
    campaign_root = Path(results_dir) / "campaigns" / campaign
    ledger_path = Path(results_dir) / "benchmark_ledger.jsonl"
    runs: List[BenchmarkCampaignRun] = []

    for profile in profiles:
        profiled_cfg = apply_benchmark_profile(base_cfg, profile)
        if cfg_overrides:
            profiled_cfg = deep_merge_dicts(profiled_cfg, cfg_overrides)
        for ablation in ablations:
            ablated_cfg = apply_benchmark_ablation(deepcopy(profiled_cfg), ablation)
            for repeat_index in range(1, repeat + 1):
                run_dir = campaign_root / _slug(profile.name) / _slug(ablation.name) / f"repeat_{repeat_index:02d}"
                run_cfg = deepcopy(ablated_cfg)
                run_cfg.setdefault("runtime", {})
                run_cfg["runtime"]["event_log_path"] = str(run_dir / "runtime_events.jsonl")
                seed = int(run_cfg.get("seed", 1337))
                device = configure_runtime(
                    run_cfg,
                    deterministic_override=deterministic_override,
                    safe_override=safe_override,
                )
                event_logger = build_runtime_event_logger(run_cfg)
                prover, tokenizer, verifier = load_benchmark_runtime(run_cfg, device, checkpoint=checkpoint)
                model_runtime = build_model_runtime_info(run_cfg, prover)

                suite_results: List[BenchmarkSuiteResult] = []
                for target_kind, target_name in targets:
                    result = run_suite_target(
                        target_kind,
                        target_name,
                        run_cfg,
                        prover,
                        verifier,
                        tokenizer,
                        device,
                        checker_plugin,
                        event_logger,
                        max_cases=max_cases,
                    )
                    result.metadata.update(
                        {
                            "campaign": campaign,
                            "profile": profile.name,
                            "ablation": ablation.name,
                            "repeat_index": repeat_index,
                            "model_runtime": model_runtime,
                            "git": git_context,
                        }
                    )
                    save_suite_result(str(run_dir), result)
                    suite_results.append(result)

                run_record = BenchmarkCampaignRun(
                    profile=profile.name,
                    ablation=ablation.name,
                    repeat_index=repeat_index,
                    seed=seed,
                    suite_results=suite_results,
                    model_runtime=model_runtime,
                    git=git_context,
                    config=run_cfg,
                    metadata={
                        "profile_tags": list(profile.tags),
                        "ablation_tags": list(ablation.tags),
                        "profile_metadata": dict(profile.metadata),
                        "report_lane": str(
                            run_cfg.get("benchmark", {}).get("report_lane", profile.metadata.get("purpose", "unspecified"))
                        ).strip()
                        or "unspecified",
                    },
                )
                save_json(str(run_dir / "config_snapshot.json"), run_cfg)
                save_json(
                    str(run_dir / "manifest.json"),
                    {
                        "campaign": campaign,
                        "profile": profile.to_dict(),
                        "ablation": ablation.to_dict(),
                        "repeat_index": repeat_index,
                        "suite_spec": suite_spec,
                        "targets": [f"{kind}:{name}" for kind, name in targets],
                        "max_cases": max_cases,
                        "git": git_context,
                        "model_runtime": model_runtime,
                    },
                )
                save_json(str(run_dir / "run_summary.json"), run_record.to_dict())
                append_jsonl(
                    str(ledger_path),
                    {
                        "campaign": campaign,
                        "profile": profile.name,
                        "ablation": ablation.name,
                        "repeat_index": repeat_index,
                        "aggregate": run_record.aggregate_metrics(),
                        "git": git_context,
                        "model_runtime": model_runtime,
                    },
                )
                runs.append(run_record)

    summary = BenchmarkCampaignSummary(
        name=campaign,
        suite_spec=suite_spec,
        targets=[f"{kind}:{name}" for kind, name in targets],
        repeat=repeat,
        runs=runs,
        variants=summarize_campaign_runs(runs),
        metadata={
            "git": git_context,
            "profile_names": [profile.name for profile in profiles],
            "ablation_names": [ablation.name for ablation in ablations],
            "max_cases": max_cases,
            "runtime_fingerprints": sorted(
                {
                    json.dumps(
                        {
                            "provider": run.model_runtime.get("provider", ""),
                            "backbone": run.model_runtime.get("backbone", ""),
                            "adapter_mode": run.model_runtime.get("adapter_mode", ""),
                            "quantization": run.model_runtime.get("quantization", ""),
                        },
                        sort_keys=True,
                    )
                    for run in runs
                }
            ),
        },
    )
    save_json(str(campaign_root / "campaign_summary.json"), summary.to_dict())
    save_text(str(campaign_root / "campaign_report.md"), render_campaign_report(summary))
    return summary
