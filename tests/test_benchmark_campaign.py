from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.base import BenchmarkSuiteResult
from benchmarks.campaign import run_benchmark_campaign
from benchmarks.profiles import resolve_benchmark_profiles
from benchmarks.ablations import resolve_benchmark_ablations
from sentinel.config import load_runtime_config


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"


class BenchmarkCampaignTests(unittest.TestCase):
    def _fresh_dir(self, name: str) -> Path:
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        path = TEST_TMP_ROOT / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    @patch("benchmarks.campaign.run_suite_target")
    @patch("benchmarks.campaign.load_benchmark_runtime")
    def test_campaign_writes_summary_report_and_ledger(self, mock_load_runtime: object, mock_run_suite_target: object) -> None:
        mock_load_runtime.return_value = (object(), object(), object())
        seen_max_cases: list[object] = []

        def fake_run_suite_target(target_kind: str, target_name: str, *args: object, **kwargs: object) -> BenchmarkSuiteResult:  # noqa: ARG001
            seen_max_cases.append(kwargs.get("max_cases"))
            backend = "swebench_ops" if target_name == "swebench_verified_smoke" else "gaia_ops" if target_name == "gaia_smoke" else "math"
            return BenchmarkSuiteResult(
                suite=target_name if target_kind == "public" else f"internal_{target_name}",
                backend=backend,
                tier="smoke",
                description="fake benchmark result",
                solved_rate=1.0,
                equivalence_rate=1.0,
                avg_branches=2.0,
                cases=[],
                metadata={},
            )

        mock_run_suite_target.side_effect = fake_run_suite_target
        results_dir = self._fresh_dir("benchmark-campaign")
        base_cfg = load_runtime_config("config/benchmarks/public_smoke.yaml", search_config_path="")

        summary = run_benchmark_campaign(
            base_cfg=base_cfg,
            cfg_overrides=None,
            suite_spec="public_smoke",
            backends_spec="all",
            profiles=resolve_benchmark_profiles("smoke_tiny"),
            ablations=resolve_benchmark_ablations("baseline,no_retrieval"),
            checkpoint="",
            results_dir=str(results_dir),
            checker_plugin="",
            deterministic_override=True,
            safe_override=True,
            repeat=2,
            campaign_name="unit_campaign",
            max_cases=7,
        )

        campaign_root = results_dir / "campaigns" / "unit_campaign"
        ledger_path = results_dir / "benchmark_ledger.jsonl"

        self.assertEqual(summary.name, "unit_campaign")
        self.assertEqual(len(summary.variants), 2)
        self.assertTrue(all("lane" in variant for variant in summary.variants))
        self.assertTrue(all("integrity_pass_rate" in variant["aggregate"] for variant in summary.variants))
        self.assertTrue((campaign_root / "campaign_summary.json").exists())
        self.assertTrue((campaign_root / "campaign_report.md").exists())
        self.assertEqual(len([line for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]), 4)
        self.assertTrue(all(variant["stable"] for variant in summary.variants))
        self.assertEqual(seen_max_cases, [7] * 12)
        self.assertEqual(summary.metadata.get("max_cases"), 7)

    @patch("benchmarks.campaign.run_suite_target")
    @patch("benchmarks.campaign.load_benchmark_runtime")
    def test_campaign_applies_cfg_overrides_after_profile(self, mock_load_runtime: object, mock_run_suite_target: object) -> None:
        mock_load_runtime.return_value = (object(), object(), object())

        def fake_run_suite_target(target_kind: str, target_name: str, cfg: dict, *args: object, **kwargs: object) -> BenchmarkSuiteResult:  # noqa: ARG001
            self.assertEqual(cfg["model"]["backbone"], "models/Qwen2.5-Coder-1.5B-Instruct")
            self.assertTrue(cfg["model"]["local_files_only"])
            return BenchmarkSuiteResult(
                suite=target_name if target_kind == "public" else f"internal_{target_name}",
                backend="swebench_ops",
                tier="smoke",
                description="fake benchmark result",
                solved_rate=1.0,
                equivalence_rate=1.0,
                avg_branches=1.0,
                cases=[],
                metadata={},
            )

        mock_run_suite_target.side_effect = fake_run_suite_target
        results_dir = self._fresh_dir("benchmark-campaign-overrides")
        base_cfg = load_runtime_config("config/benchmarks/public_smoke.yaml", search_config_path="")

        run_benchmark_campaign(
            base_cfg=base_cfg,
            cfg_overrides={"model": {"backbone": "models/Qwen2.5-Coder-1.5B-Instruct", "local_files_only": True}},
            suite_spec="swebench_verified_smoke",
            backends_spec="all",
            profiles=resolve_benchmark_profiles("rtx4060_coder_local"),
            ablations=resolve_benchmark_ablations("baseline"),
            checkpoint="",
            results_dir=str(results_dir),
            checker_plugin="",
            deterministic_override=True,
            safe_override=True,
            repeat=1,
            campaign_name="unit_campaign_overrides",
        )
