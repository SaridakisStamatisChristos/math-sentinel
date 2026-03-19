from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from benchmarks.manifest_loader import load_manifest_suite
from benchmarks.official_corpus import (
    ensure_official_manifest,
    official_corpus_preflight,
    prepare_selected_official_corpora,
    resolve_official_corpus_selection,
    resolve_official_corpus_specs,
)
from sentinel.config import load_runtime_config


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"


class OfficialCorpusWorkflowTests(unittest.TestCase):
    def _fresh_dir(self, name: str) -> Path:
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        path = TEST_TMP_ROOT / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def _write_gaia_fixture(self, root: Path) -> tuple[Path, Path]:
        attachments = root / "gaia" / "attachments" / "roadmap_case"
        attachments.mkdir(parents=True, exist_ok=True)
        (attachments / "roadmap.json").write_text(
            json.dumps({"tasks": [{"owner": "Mira", "task": "Finalize budget", "due": "2026-03-20"}]}),
            encoding="utf-8",
        )
        records_path = root / "gaia" / "records.jsonl"
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records_path.write_text(
            json.dumps(
                {
                    "question_id": "roadmap_case",
                    "question": "Which pending task owned by Mira has the earliest due date?",
                    "final_answer": "Finalize budget",
                    "fixture_dir": str(attachments),
                    "oracle_evidence_file": "roadmap.json",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return records_path, attachments.parent

    def _write_swebench_fixture(self, root: Path) -> tuple[Path, Path]:
        workspace = root / "swebench" / "workspaces" / "count_bug"
        tests_dir = workspace / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        (workspace / "stats.py").write_text("def count_positive(values):\n    return sum(1 for value in values if value >= 0)\n", encoding="utf-8")
        (tests_dir / "test_stats.py").write_text(
            "import unittest\nfrom stats import count_positive\n\nclass StatsTests(unittest.TestCase):\n    def test_count_positive(self):\n        self.assertEqual(count_positive([-1, 0, 2]), 1)\n\nif __name__ == '__main__':\n    unittest.main()\n",
            encoding="utf-8",
        )
        records_path = root / "swebench" / "records.jsonl"
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records_path.write_text(
            json.dumps(
                {
                    "instance_id": "count_bug",
                    "problem_statement": "Patch the repository so the tests pass.",
                    "fixture_dir": str(workspace),
                    "oracle_primary_file": "stats.py",
                    "test_command": ["python", "-m", "unittest", "discover", "-s", "tests", "-q"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return records_path, workspace.parent

    def _config_with_overrides(self, temp_root: Path, gaia_records: Path, gaia_root: Path, swebench_records: Path, swebench_root: Path) -> Path:
        manifest_root = temp_root / "manifests"
        config_path = temp_root / "official_corpus_test.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "extends: ../../config/default.yaml",
                    "official_corpus:",
                    "  gaia:",
                    f"    input_path: {gaia_records.as_posix()}",
                    f"    fixtures_root: {gaia_root.as_posix()}",
                    f"    manifest_path: {(manifest_root / 'gaia_full_official.json').as_posix()}",
                    "    suite_name: gaia_full_official_test",
                    "  swebench:",
                    f"    input_path: {swebench_records.as_posix()}",
                    f"    fixtures_root: {swebench_root.as_posix()}",
                    f"    manifest_path: {(manifest_root / 'swebench_full_official.json').as_posix()}",
                    "    suite_name: swebench_full_official_test",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return config_path

    def test_default_official_specs_exist(self) -> None:
        specs = resolve_official_corpus_specs({})

        self.assertIn("gaia", specs)
        self.assertIn("swebench", specs)
        self.assertTrue(specs["gaia"].manifest_path.endswith("gaia_full_official.json"))
        self.assertEqual(specs["swebench"].default_profile, "public_claim_coder_local_1p5b")

    def test_official_preflight_reports_missing_data(self) -> None:
        temp_root = self._fresh_dir("official-corpus-missing")
        cfg = {
            "official_corpus": {
                "gaia": {
                    "input_path": str((temp_root / "gaia" / "records.jsonl").resolve()),
                    "fixtures_root": str((temp_root / "gaia" / "attachments").resolve()),
                },
                "swebench": {
                    "input_path": str((temp_root / "swebench" / "records.jsonl").resolve()),
                    "fixtures_root": str((temp_root / "swebench" / "workspaces").resolve()),
                },
            }
        }
        report = official_corpus_preflight(cfg)

        self.assertEqual({item["name"] for item in report}, {"gaia", "swebench"})
        self.assertTrue(any(not bool(item["input_exists"]) for item in report))

    def test_prepare_selected_official_corpora_builds_manifests(self) -> None:
        temp_root = self._fresh_dir("official-corpus-prepare")
        gaia_records, gaia_root = self._write_gaia_fixture(temp_root)
        swebench_records, swebench_root = self._write_swebench_fixture(temp_root)
        config_path = self._config_with_overrides(temp_root, gaia_records, gaia_root, swebench_records, swebench_root)
        cfg = load_runtime_config(str(config_path), search_config_path="")

        prepared = prepare_selected_official_corpora(cfg, "all", strict_materialization=True)

        self.assertEqual(len(prepared), 2)
        gaia_manifest = next(item["manifest_path"] for item in prepared if item["name"] == "gaia")
        swebench_manifest = next(item["manifest_path"] for item in prepared if item["name"] == "swebench")
        self.assertTrue(Path(gaia_manifest).exists())
        self.assertTrue(Path(swebench_manifest).exists())
        self.assertEqual(load_manifest_suite(gaia_manifest).name, "gaia_full_official_test")
        self.assertEqual(load_manifest_suite(swebench_manifest).name, "swebench_full_official_test")

    def test_ensure_official_manifest_returns_existing_manifest(self) -> None:
        temp_root = self._fresh_dir("official-corpus-ensure")
        gaia_records, gaia_root = self._write_gaia_fixture(temp_root)
        swebench_records, swebench_root = self._write_swebench_fixture(temp_root)
        config_path = self._config_with_overrides(temp_root, gaia_records, gaia_root, swebench_records, swebench_root)
        cfg = load_runtime_config(str(config_path), search_config_path="")

        manifest_path = ensure_official_manifest("gaia", cfg, strict_materialization=True)

        self.assertTrue(Path(manifest_path).exists())
        again = ensure_official_manifest("gaia", cfg, strict_materialization=True)
        self.assertEqual(Path(manifest_path).resolve(), Path(again).resolve())

    def test_resolve_official_corpus_selection_validates_names(self) -> None:
        self.assertEqual(resolve_official_corpus_selection("gaia"), ["gaia"])
        self.assertEqual(resolve_official_corpus_selection("all"), ["gaia", "swebench"])
        with self.assertRaises(ValueError):
            resolve_official_corpus_selection("unknown")

    def test_run_official_corpus_cli_prepare_only(self) -> None:
        temp_root = self._fresh_dir("official-corpus-cli")
        gaia_records, gaia_root = self._write_gaia_fixture(temp_root)
        swebench_records, swebench_root = self._write_swebench_fixture(temp_root)
        config_path = self._config_with_overrides(temp_root, gaia_records, gaia_root, swebench_records, swebench_root)

        proc = subprocess.run(
            [
                sys.executable,
                "benchmarks/run_official_corpus.py",
                "--config",
                str(config_path),
                "--corpus",
                "all",
                "--prepare-only",
                "--strict-materialization",
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=True,
        )

        payload = json.loads(proc.stdout)
        self.assertEqual(len(payload["prepared"]), 2)
        self.assertTrue(all(bool(item["prepared"]) for item in payload["prepared"]))


if __name__ == "__main__":
    unittest.main()
