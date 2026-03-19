#!/usr/bin/env python3
from __future__ import annotations

import argparse

from benchmarks.official_ingest import ingest_gaia_records, ingest_swebench_records


def main() -> None:
    ap = argparse.ArgumentParser(description="Import local public-benchmark exports into Math Sentinel manifest suites")
    ap.add_argument("--format", choices=["swebench", "gaia"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fixtures-root", default="")
    ap.add_argument("--suite-name", default="")
    ap.add_argument("--tier", default="official")
    ap.add_argument("--description", default="")
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

    print(written)


if __name__ == "__main__":
    main()
