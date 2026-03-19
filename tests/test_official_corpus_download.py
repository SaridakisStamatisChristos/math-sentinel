from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from benchmarks.download_official_corpus import normalize_gaia_record, normalize_swebench_record
from benchmarks.manifest_loader import lint_manifest_suite
from domains.repo_agent_utils import create_workspace


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"


class _Task:
    def __init__(self, task_id: str, prompt: str, meta: dict[str, object]) -> None:
        self.task_id = task_id
        self.prompt = prompt
        self.meta = meta


class OfficialCorpusDownloadTests(unittest.TestCase):
    def _fresh_dir(self, name: str) -> Path:
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        path = TEST_TMP_ROOT / f"{name}-{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def test_normalize_swebench_record_preserves_repo_metadata(self) -> None:
        record = normalize_swebench_record(
            {
                "repo": "psf/requests",
                "instance_id": "psf__requests-1",
                "base_commit": "abc123",
                "problem_statement": "Fix the bug",
                "test_patch": "diff --git a/tests/test_x.py b/tests/test_x.py",
            }
        )

        self.assertEqual(record["repo"], "psf/requests")
        self.assertEqual(record["repo_clone_url"], "https://github.com/psf/requests.git")
        self.assertEqual(record["instance_id"], "psf__requests-1")
        self.assertEqual(record["base_commit"], "abc123")

    def test_normalize_gaia_record_handles_capitalized_columns(self) -> None:
        record = normalize_gaia_record(
            {
                "Question": "What is the answer?",
                "Final answer": "42",
                "Level": 2,
                "file_name": "evidence.csv",
                "Question_ID": "gaia_case_1",
            }
        )

        self.assertEqual(record["question_id"], "gaia_case_1")
        self.assertEqual(record["question"], "What is the answer?")
        self.assertEqual(record["final_answer"], "42")
        self.assertEqual(record["file_name"], "evidence.csv")

    def test_manifest_lint_accepts_swebench_repo_materialization_metadata(self) -> None:
        temp_root = self._fresh_dir("official-corpus-manifest")
        manifest = temp_root / "swebench_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "name": "swebench_repo_materialized",
                    "backend": "swebench_ops",
                    "cases": [
                        {
                            "task_id": "requests__bug-1",
                            "domain": "swebench_patch",
                            "prompt": "Fix the repository",
                            "answer": "patched_and_verified",
                            "meta": {
                                "repo": "psf/requests",
                                "base_commit": "abc123",
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        report = lint_manifest_suite(str(manifest), strict_materialization=True)

        self.assertTrue(report["valid"])

    @patch("domains.repo_agent_utils._apply_git_patch")
    @patch("domains.repo_agent_utils._materialize_repo_workspace")
    def test_create_workspace_clones_local_repo_and_applies_test_patch(self, mock_materialize: object, mock_apply_patch: object) -> None:
        temp_root = self._fresh_dir("official-corpus-workspace")
        base_commit = "abc123"
        patch_text = "diff --git a/tests/test_stats.py b/tests/test_stats.py"

        def _fake_materialize(repo_slug: str, commit: str, workspace: Path, *, repo_cache_root: Path, clone_url: str = "") -> None:
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "stats.py").write_text("def count_positive(values):\n    return 1\n", encoding="utf-8")

        mock_materialize.side_effect = _fake_materialize

        task = _Task(
            "requests__bug-1",
            "Fix the repository",
            {
                "repo": "local/repo",
                "repo_clone_url": "file:///tmp/origin_repo",
                "base_commit": base_commit,
                "test_patch": patch_text,
                "repo_cache_root": str(temp_root / "repo_cache"),
            },
        )

        workspace = create_workspace(task, temp_root / "workspaces", deterministic=True)

        self.assertTrue((workspace / "stats.py").exists())
        mock_materialize.assert_called_once()
        mock_apply_patch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
