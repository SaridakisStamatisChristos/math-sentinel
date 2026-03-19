#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.manifest_loader import lint_manifest_suite


def main() -> None:
    ap = argparse.ArgumentParser(description="Lint a benchmark manifest before running it through Math Sentinel.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--strict-materialization", action="store_true")
    args = ap.parse_args()

    report = lint_manifest_suite(args.manifest, strict_materialization=bool(args.strict_materialization))
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if not bool(report.get("valid", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
