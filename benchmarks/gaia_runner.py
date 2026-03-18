#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import save_suite_result
from benchmarks.runners import load_benchmark_runtime, run_public_suite
from sentinel.config import load_runtime_config
from sentinel.runtime import configure_runtime
from sentinel.runtime_events import build_runtime_event_logger


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the local GAIA-style public smoke suite")
    ap.add_argument("--config", default="config/benchmarks/public_smoke.yaml")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--model-provider", default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--safe-runtime", action="store_true")
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
    result = run_public_suite("gaia_smoke", cfg, prover, verifier, tokenizer, device, args.checker_plugin, event_logger)
    save_suite_result(args.results_dir, result)
    print(result.to_dict())


if __name__ == "__main__":
    main()
