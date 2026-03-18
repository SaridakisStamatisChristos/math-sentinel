#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any, Dict

from benchmarks import available_public_suites, save_suite_result
from benchmarks.runners import load_benchmark_runtime, resolve_backends, run_backend_benchmark, run_public_suite
from sentinel.config import load_runtime_config
from sentinel.runtime import configure_runtime
from sentinel.runtime_events import build_runtime_event_logger


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark Math Sentinel V7 backends")
    ap.add_argument("--config", default="config/default.yaml")
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
    prover, tokenizer, verifier = load_benchmark_runtime(cfg, device, checkpoint=args.checkpoint)

    normalized_suite = args.suite.strip().lower()
    if normalized_suite in {"internal", "all"}:
        for backend_name in resolve_backends(args.backends):
            result = run_backend_benchmark(backend_name, cfg, prover, verifier, tokenizer, device, args.checker_plugin, event_logger)
            save_suite_result(args.results_dir, result)
            print(result.to_dict())

    if normalized_suite in {"public_smoke", "all"}:
        for suite_name in available_public_suites():
            result = run_public_suite(suite_name, cfg, prover, verifier, tokenizer, device, args.checker_plugin, event_logger)
            save_suite_result(args.results_dir, result)
            print(result.to_dict())

    if normalized_suite in set(available_public_suites()):
        result = run_public_suite(normalized_suite, cfg, prover, verifier, tokenizer, device, args.checker_plugin, event_logger)
        save_suite_result(args.results_dir, result)
        print(result.to_dict())


if __name__ == "__main__":
    main()
