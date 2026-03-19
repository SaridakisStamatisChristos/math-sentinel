#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.official_ingest import ingest_gaia_records, ingest_swebench_records
from benchmarks.manifest_loader import lint_manifest_suite


def main() -> None:
    ap = argparse.ArgumentParser(description="Import local public-benchmark exports into Math Sentinel manifest suites")
    ap.add_argument("--format", choices=["swebench", "gaia"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fixtures-root", default="")
    ap.add_argument("--suite-name", default="")
    ap.add_argument("--tier", default="official")
    ap.add_argument("--description", default="")
    ap.add_argument("--strict-materialization", action="store_true")
    args = ap.parse_args()

    if args.format == "swebench":
        written = ingest_swebench_records(
            args.input,
            args.output,
            fixtures_root=args.fixtures_root,
            suite_name=args.suite_name or "swebench_verified_public_import",
            tier=args.tier,
            description=args.description or "Official-style SWE-bench manifest imported from local records.",
        )
    else:
        written = ingest_gaia_records(
            args.input,
            args.output,
            fixtures_root=args.fixtures_root,
            suite_name=args.suite_name or "gaia_public_import",
            tier=args.tier,
            description=args.description or "Official-style GAIA manifest imported from local records.",
        )

    report = lint_manifest_suite(written, strict_materialization=bool(args.strict_materialization))
    print(json.dumps({"written": written, "lint": report}, ensure_ascii=True, indent=2))
    if not bool(report.get("valid", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
