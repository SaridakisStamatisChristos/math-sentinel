from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from benchmarks.download_official_corpus import normalize_gaia_record, normalize_swebench_record
from benchmarks.manifest_loader import lint_manifest_suite
from domains.repo_agent_utils import (
    _apply_unified_patch_sections,
    _ensure_repo_cache,
    _materialize_repo_workspace,
    _recount_unified_patch,
    create_workspace,
)


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

    def test_recount_unified_patch_repairs_hunk_counts(self) -> None:
        patch = "\n".join(
            [
                "diff --git a/file.txt b/file.txt",
                "--- a/file.txt",
                "+++ b/file.txt",
                "@@ -2,7 +2,9 @@",
                " line2",
                "-line3",
                "+line3 changed",
                "+line3 extra",
                " line4",
                "",
            ]
        )

        rebuilt = _recount_unified_patch(patch)

        self.assertIn("@@ -2,3 +2,4 @@", rebuilt)

    def test_apply_unified_patch_sections_updates_workspace_file(self) -> None:
        temp_root = self._fresh_dir("official-corpus-patch")
        workspace = temp_root / "workspace"
        target = workspace / "pkg" / "example.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("line1\nline2\nline3\n", encoding="utf-8")
        patch = "\n".join(
            [
                "diff --git a/pkg/example.py b/pkg/example.py",
                "--- a/pkg/example.py",
                "+++ b/pkg/example.py",
                "@@ -2,2 +2,3 @@",
                " line2",
                "-line3",
                "+line3 changed",
                "+line4",
                "",
            ]
        )

        touched = _apply_unified_patch_sections(workspace, patch)

        self.assertEqual(touched, ["pkg/example.py"])
        self.assertEqual(target.read_text(encoding="utf-8"), "line1\nline2\nline3 changed\nline4\n")

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

    @patch("domains.repo_agent_utils._run_git")
    @patch("domains.repo_agent_utils._git_has_commit", return_value=True)
    def test_ensure_repo_cache_skips_fetch_when_required_commit_exists(self, mock_has_commit: object, mock_run_git: object) -> None:
        temp_root = self._fresh_dir("official-corpus-cache")
        cache_repo = temp_root / "psf__requests"
        cache_repo.mkdir(parents=True, exist_ok=True)

        resolved = _ensure_repo_cache("psf/requests", temp_root, required_commit="abc123")

        self.assertEqual(resolved, cache_repo)
        mock_has_commit.assert_called_once()
        mock_run_git.assert_not_called()

    @patch("domains.repo_agent_utils._rebuild_repo_cache")
    @patch("domains.repo_agent_utils._repo_is_partial_clone", return_value=True)
    @patch("domains.repo_agent_utils._export_repo_workspace")
    @patch("domains.repo_agent_utils._ensure_repo_cache")
    def test_materialize_repo_workspace_rebuilds_partial_clone_when_promisor_export_fails(
        self,
        mock_ensure_cache: object,
        mock_export: object,
        mock_partial_clone: object,
        mock_rebuild: object,
    ) -> None:
        temp_root = self._fresh_dir("official-corpus-materialize")
        cache_repo = temp_root / "cache"
        rebuilt_repo = temp_root / "cache_rebuilt"
        cache_repo.mkdir(parents=True, exist_ok=True)
        rebuilt_repo.mkdir(parents=True, exist_ok=True)
        workspace = temp_root / "workspace"
        mock_ensure_cache.return_value = cache_repo
        mock_rebuild.return_value = rebuilt_repo
        mock_export.side_effect = [
            RuntimeError("fatal: could not fetch abc from promisor remote"),
            None,
        ]

        _materialize_repo_workspace("psf/requests", "abc123", workspace, repo_cache_root=temp_root)

        mock_rebuild.assert_called_once()
        self.assertEqual(mock_export.call_count, 2)
        first_args = mock_export.call_args_list[0].args
        second_args = mock_export.call_args_list[1].args
        self.assertEqual(first_args[0], cache_repo)
        self.assertEqual(second_args[0], rebuilt_repo)


if __name__ == "__main__":
    unittest.main()
