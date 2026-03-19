#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.official_corpus import (  # noqa: E402
    default_profile_for_official_corpus,
    ensure_official_manifest,
    official_corpus_preflight,
    resolve_official_corpus_selection,
)
from benchmarks.profiles import apply_benchmark_profile, resolve_benchmark_profiles  # noqa: E402
from benchmarks.results import save_json, save_suite_result  # noqa: E402
from benchmarks.runners import load_benchmark_runtime, run_manifest_suite  # noqa: E402
from sentinel.config import load_runtime_config  # noqa: E402
from sentinel.runtime import configure_runtime  # noqa: E402
from sentinel.runtime_events import build_runtime_event_logger  # noqa: E402


def _choose_profile_name(corpus_name: str, cfg: Dict[str, Any], override: str) -> str:
    if override:
        return override
    return default_profile_for_official_corpus(corpus_name, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare and run full official corpora through Math Sentinel.")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--search-config", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--corpus", default="all")
    ap.add_argument("--profile", default="")
    ap.add_argument("--profiles-config", default="config/benchmarks/profiles.yaml")
    ap.add_argument("--results-dir", default="results/official")
    ap.add_argument("--checker-plugin", default="")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--strict-materialization", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--safe-runtime", action="store_true")
    args = ap.parse_args()

    base_cfg = load_runtime_config(args.config, search_config_path=args.search_config)
    selected = resolve_official_corpus_selection(args.corpus, base_cfg)
    preflight = official_corpus_preflight(base_cfg)
    preflight_map = {item["name"]: item for item in preflight}

    if args.prepare_only:
        prepared: List[Dict[str, Any]] = []
        for corpus_name in selected:
            manifest_path = ensure_official_manifest(
                corpus_name,
                base_cfg,
                strict_materialization=(True if args.strict_materialization else None),
            )
            status = dict(preflight_map.get(corpus_name, {}))
            status["manifest_path"] = manifest_path
            status["prepared"] = True
            prepared.append(status)
        print(json.dumps({"preflight": preflight, "prepared": prepared}, ensure_ascii=True, indent=2))
        return

    summaries: List[Dict[str, Any]] = []
    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    for corpus_name in selected:
        manifest_path = ensure_official_manifest(
            corpus_name,
            base_cfg,
            strict_materialization=(True if args.strict_materialization else None),
        )
        profile_name = _choose_profile_name(corpus_name, base_cfg, args.profile)
        profile = resolve_benchmark_profiles(profile_name, args.profiles_config)[0]
        cfg = apply_benchmark_profile(base_cfg, profile)
        device = configure_runtime(
            cfg,
            deterministic_override=(True if args.deterministic else None),
            safe_override=(True if args.safe_runtime else None),
        )
        event_logger = build_runtime_event_logger(cfg)
        prover, tokenizer, verifier = load_benchmark_runtime(cfg, device, checkpoint=args.checkpoint)
        result = run_manifest_suite(
            manifest_path,
            cfg,
            prover,
            verifier,
            tokenizer,
            device,
            args.checker_plugin,
            event_logger,
        )
        result_path = save_suite_result(str(results_root), result)
        summary = {
            "corpus": corpus_name,
            "profile": profile_name,
            "manifest_path": manifest_path,
            "result_path": result_path,
            "result": result.to_dict(),
        }
        summaries.append(summary)

    summary_path = save_json(str(results_root / "official_corpus_summary.json"), {"runs": summaries, "preflight": preflight})
    print(json.dumps({"summary_path": summary_path, "runs": summaries}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
