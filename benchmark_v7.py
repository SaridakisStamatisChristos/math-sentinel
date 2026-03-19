#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any, Dict

from benchmarks.ablations import resolve_benchmark_ablations
from benchmarks.ablations import available_benchmark_ablations
from benchmarks.campaign import run_benchmark_campaign
from benchmarks.profiles import apply_benchmark_profile, resolve_benchmark_profiles
from benchmarks.profiles import available_benchmark_profiles
from benchmarks.results import save_suite_result
from benchmarks.runners import load_benchmark_runtime, resolve_suite_targets, run_suite_target
from sentinel.config import load_runtime_config
from sentinel.runtime import configure_runtime
from sentinel.runtime_events import build_runtime_event_logger


def default_campaign_profile_for_suite(suite_spec: str) -> str:
    normalized = (suite_spec or "").strip().lower()
    if "swebench" in normalized:
        return "public_claim_coder_local_1p5b"
    if normalized.startswith("public") or normalized.startswith("manifest:"):
        return "public_claim_no_repairs"
    return "smoke_tiny"


def cli_cfg_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.model_provider is not None:
        overrides.setdefault("model", {})["provider"] = args.model_provider
    if args.backbone is not None:
        overrides.setdefault("model", {})["backbone"] = args.backbone
    if args.local_files_only:
        overrides.setdefault("model", {})["local_files_only"] = True
    return overrides


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark Math Sentinel V7 backends")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--search-config", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--model-provider", default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--safe-runtime", action="store_true")
    ap.add_argument("--suite", default="internal")
    ap.add_argument("--backends", default="all")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--checker-plugin", default="")
    ap.add_argument("--profile", default="")
    ap.add_argument("--profiles-config", default="config/benchmarks/profiles.yaml")
    ap.add_argument("--list-profiles", action="store_true")
    ap.add_argument("--ablations", default="")
    ap.add_argument("--ablation-config", default="config/benchmarks/ablation_matrix.yaml")
    ap.add_argument("--list-ablations", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--campaign-name", default="")
    args = ap.parse_args()

    if args.list_profiles:
        for name in available_benchmark_profiles(args.profiles_config):
            print(name)
        return

    if args.list_ablations:
        for name in available_benchmark_ablations(args.ablation_config):
            print(name)
        return

    cfg = load_runtime_config(args.config, search_config_path=args.search_config)
    overrides = cli_cfg_overrides(args)
    targets = resolve_suite_targets(args.suite, args.backends)
    campaign_mode = bool(args.profile or args.ablations or args.repeat > 1 or args.campaign_name)
    auto_profile = ""
    if not campaign_mode and any(kind in {"public", "manifest"} for kind, _ in targets):
        auto_profile = default_campaign_profile_for_suite(args.suite)
    if auto_profile:
        cfg = apply_benchmark_profile(cfg, resolve_benchmark_profiles(auto_profile, args.profiles_config)[0])
    if overrides:
        for section, values in overrides.items():
            cfg.setdefault(section, {}).update(values)

    if campaign_mode:
        profiles = resolve_benchmark_profiles(args.profile or default_campaign_profile_for_suite(args.suite), args.profiles_config)
        ablations = resolve_benchmark_ablations(args.ablations or "baseline", args.ablation_config)
        summary = run_benchmark_campaign(
            base_cfg=cfg,
            cfg_overrides=overrides,
            suite_spec=args.suite,
            backends_spec=args.backends,
            profiles=profiles,
            ablations=ablations,
            checkpoint=args.checkpoint,
            results_dir=args.results_dir,
            checker_plugin=args.checker_plugin,
            deterministic_override=(True if args.deterministic else None),
            safe_override=(True if args.safe_runtime else None),
            repeat=max(1, int(args.repeat)),
            campaign_name=args.campaign_name,
        )
        print(summary.to_dict())
        return

    device = configure_runtime(cfg, deterministic_override=(True if args.deterministic else None), safe_override=(True if args.safe_runtime else None))
    event_logger = build_runtime_event_logger(cfg)
    prover, tokenizer, verifier = load_benchmark_runtime(cfg, device, checkpoint=args.checkpoint)

    for target_kind, target_name in targets:
        result = run_suite_target(target_kind, target_name, cfg, prover, verifier, tokenizer, device, args.checker_plugin, event_logger)
        save_suite_result(args.results_dir, result)
        print(result.to_dict())


if __name__ == "__main__":
    main()
