from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, List, Tuple

from sentinel.model_backends import build_model_runtime_info
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
        return {
            "solved_rate": solved_rate,
            "equivalence_rate": equivalence_rate,
            "avg_branches": avg_branches,
        }

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
        aggregate_solved = _mean([run.aggregate_metrics()["solved_rate"] for run in group])
        aggregate_equivalence = _mean([run.aggregate_metrics()["equivalence_rate"] for run in group])
        aggregate_branches = _mean([run.aggregate_metrics()["avg_branches"] for run in group])
        stable = len(
            {
                (
                    round(run.aggregate_metrics()["solved_rate"], 6),
                    round(run.aggregate_metrics()["equivalence_rate"], 6),
                    round(run.aggregate_metrics()["avg_branches"], 6),
                )
                for run in group
            }
        ) == 1
        suite_breakdown = []
        for suite_name, suite_runs in sorted(suite_groups.items()):
            suite_breakdown.append(
                {
                    "suite": suite_name,
                    "solved_rate": _mean([entry.solved_rate for entry in suite_runs]),
                    "equivalence_rate": _mean([entry.equivalence_rate for entry in suite_runs]),
                    "avg_branches": _mean([entry.avg_branches for entry in suite_runs]),
                }
            )
        variants.append(
            {
                "profile": profile,
                "ablation": ablation,
                "repeats": len(group),
                "stable": stable,
                "aggregate": {
                    "solved_rate": aggregate_solved,
                    "equivalence_rate": aggregate_equivalence,
                    "avg_branches": aggregate_branches,
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
        "",
        "## Variants",
        "",
    ]
    for variant in summary.variants:
        aggregate = variant["aggregate"]
        delta = variant["delta_vs_baseline"]
        lines.append(
            (
                f"- `{variant['profile']}/{variant['ablation']}` "
                f"solved={aggregate['solved_rate']:.3f} "
                f"equiv={aggregate['equivalence_rate']:.3f} "
                f"branches={aggregate['avg_branches']:.3f} "
                f"stable={variant['stable']} "
                f"delta_equiv={delta['equivalence_rate']:+.3f}"
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
                    f"branches={suite['avg_branches']:.3f}"
                )
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def run_benchmark_campaign(
    *,
    base_cfg: Dict[str, Any],
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
        },
    )
    save_json(str(campaign_root / "campaign_summary.json"), summary.to_dict())
    save_text(str(campaign_root / "campaign_report.md"), render_campaign_report(summary))
    return summary
