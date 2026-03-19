from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.base import BenchmarkSuiteResult
from benchmarks.manifest_loader import load_manifest_suite
from benchmarks.official_ingest import ingest_gaia_records, ingest_swebench_records
from benchmarks.runners import resolve_suite_targets, run_suite_target
from sentinel.config import load_runtime_config


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"


class BenchmarkManifestTests(unittest.TestCase):
    def _fresh_dir(self, name: str) -> Path:
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        path = TEST_TMP_ROOT / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def test_load_manifest_suite_resolves_relative_fixture_paths(self) -> None:
        suite = load_manifest_suite("benchmarks/manifests/swebench_verified_medium_official_style.json")

        self.assertEqual(suite.name, "swebench_verified_medium_official_style")
        self.assertEqual(suite.backend, "swebench_ops")
        self.assertEqual(suite.tier, "official_style")
        self.assertGreaterEqual(len(suite.cases), 1)
        fixture_dir = Path(suite.cases[0].meta["fixture_dir"])
        self.assertTrue(fixture_dir.is_absolute())
        self.assertTrue(fixture_dir.exists())

    def test_resolve_suite_targets_supports_manifest_prefix(self) -> None:
        targets = resolve_suite_targets("manifest:benchmarks/manifests/gaia_medium_official_style.json", "all")

        self.assertEqual(targets, [("manifest", "benchmarks/manifests/gaia_medium_official_style.json")])

    def test_ingest_swebench_records_writes_manifest(self) -> None:
        tmp_dir = self._fresh_dir("ingest-swebench")
        input_path = tmp_dir / "records.jsonl"
        output_path = tmp_dir / "suite.json"
        input_path.write_text(
            json.dumps(
                {
                    "instance_id": "count_bug",
                    "problem_statement": "Patch the repository so the failing tests pass. Fix the counting behavior.",
                    "fixture_dir": str((Path("benchmarks/fixtures/swebench_medium_smoke/count_bug")).resolve()),
                    "oracle_primary_file": "stats.py",
                    "test_command": ["python", "-m", "unittest", "discover", "-s", "tests", "-q"],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        written = ingest_swebench_records(str(input_path), str(output_path), suite_name="swebench_import_test")
        suite = load_manifest_suite(written)

        self.assertEqual(suite.name, "swebench_import_test")
        self.assertEqual(suite.backend, "swebench_ops")
        self.assertEqual(suite.cases[0].task_id, "count_bug")
        self.assertEqual(suite.cases[0].answer, "patched_and_verified")

    def test_ingest_gaia_records_writes_manifest(self) -> None:
        tmp_dir = self._fresh_dir("ingest-gaia")
        input_path = tmp_dir / "records.jsonl"
        output_path = tmp_dir / "suite.json"
        input_path.write_text(
            json.dumps(
                {
                    "question_id": "roadmap_dates",
                    "question": "Which pending task owned by Mira has the earliest due date?",
                    "final_answer": "Finalize budget",
                    "fixture_dir": str((Path("benchmarks/fixtures/gaia_medium_smoke/roadmap_dates")).resolve()),
                    "oracle_evidence_file": "roadmap.json",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        written = ingest_gaia_records(str(input_path), str(output_path), suite_name="gaia_import_test")
        suite = load_manifest_suite(written)

        self.assertEqual(suite.name, "gaia_import_test")
        self.assertEqual(suite.backend, "gaia_ops")
        self.assertEqual(suite.cases[0].task_id, "roadmap_dates")
        self.assertEqual(suite.cases[0].answer, "Finalize budget")

    @patch("benchmarks.runners.run_task_collection")
    def test_run_suite_target_supports_manifest_suite(self, mock_run_task_collection: object) -> None:
        mock_run_task_collection.return_value = BenchmarkSuiteResult(
            suite="gaia_medium_official_style",
            backend="gaia_ops",
            tier="official_style",
            description="fake",
            solved_rate=1.0,
            equivalence_rate=1.0,
            avg_branches=1.0,
            cases=[],
            metadata={},
        )

        cfg = load_runtime_config("config/benchmarks/public_smoke.yaml", search_config_path="")
        result = run_suite_target(
            "manifest",
            "benchmarks/manifests/gaia_medium_official_style.json",
            cfg,
            prover=object(),
            verifier=object(),
            tokenizer=object(),
            device="cpu",
            checker_plugin="",
            event_logger=None,
        )

        self.assertEqual(result.suite, "gaia_medium_official_style")
        self.assertTrue(mock_run_task_collection.called)


if __name__ == "__main__":
    unittest.main()
