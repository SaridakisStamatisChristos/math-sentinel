from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from benchmarks.download_official_corpus import normalize_gaia_record, normalize_swebench_record
from benchmarks.manifest_loader import lint_manifest_suite
from domains.repo_agent_utils import (
    _apply_unified_patch_sections,
    _ensure_repo_cache,
    _workspace_repo_test_packages,
    _packages_for_missing_modules,
    _materialize_repo_workspace,
    _read_text_file,
    _recount_unified_patch,
    create_workspace,
    infer_prompt_source_hints,
    inspect_python_tests_with_context,
    summarize_test_failures,
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

    @patch("domains.repo_agent_utils._apply_git_patch")
    @patch("domains.repo_agent_utils._materialize_repo_workspace")
    def test_create_workspace_uses_fail_to_pass_for_targeted_pytest_command(self, mock_materialize: object, mock_apply_patch: object) -> None:
        temp_root = self._fresh_dir("official-corpus-targeted-tests")

        def _fake_materialize(repo_slug: str, commit: str, workspace: Path, *, repo_cache_root: Path, clone_url: str = "") -> None:
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")

        mock_materialize.side_effect = _fake_materialize

        task = _Task(
            "requests__bug-1",
            "Fix the repository",
            {
                "repo": "local/repo",
                "base_commit": "abc123",
                "FAIL_TO_PASS": json.dumps(["tests/test_bug.py::test_regression"]),
                "PASS_TO_PASS": json.dumps(["tests/test_smoke.py::test_smoke"]),
            },
        )

        workspace = create_workspace(task, temp_root / "workspaces", deterministic=True)

        self.assertTrue(workspace.exists())
        self.assertEqual(
            task.meta["test_command"],
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_bug.py::test_regression",
                "tests/test_smoke.py::test_smoke",
            ],
        )

    @patch("domains.repo_agent_utils._apply_git_patch")
    @patch("domains.repo_agent_utils._materialize_repo_workspace")
    def test_create_workspace_adds_repo_specific_pytest_args_for_astropy(self, mock_materialize: object, mock_apply_patch: object) -> None:
        temp_root = self._fresh_dir("official-corpus-astropy-pytest-args")

        def _fake_materialize(repo_slug: str, commit: str, workspace: Path, *, repo_cache_root: Path, clone_url: str = "") -> None:
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")

        mock_materialize.side_effect = _fake_materialize

        task = _Task(
            "astropy__astropy-12907",
            "Fix the repository",
            {
                "repo": "astropy/astropy",
                "base_commit": "abc123",
                "FAIL_TO_PASS": json.dumps(["astropy/modeling/tests/test_separable.py::test_separable[compound_model6-result6]"]),
            },
        )

        create_workspace(task, temp_root / "workspaces", deterministic=True)

        self.assertEqual(
            task.meta["test_command"][:6],
            [sys.executable, "-m", "pytest", "-q", "-p", "no:warnings"],
        )

    def test_read_text_file_skips_binary_payloads(self) -> None:
        temp_root = self._fresh_dir("official-corpus-binary-read")
        binary_path = temp_root / "payload.bin"
        binary_path.write_bytes(b"\x1f\x8b\x08\x00binary")

        self.assertEqual(_read_text_file(binary_path), "")

    def test_infer_prompt_source_hints_maps_issue_imports_to_workspace_files(self) -> None:
        files = [
            "astropy/modeling/separable.py",
            "astropy/modeling/tests/test_separable.py",
            "astropy/modeling/__init__.py",
        ]
        prompt = (
            "Fix the bug in separability_matrix.\n"
            "```python\n"
            "from astropy.modeling.separable import separability_matrix\n"
            "```\n"
        )

        hints = infer_prompt_source_hints(prompt, files)

        self.assertIn("astropy/modeling/separable.py", hints["candidate_source_files"])
        self.assertIn("separability_matrix", hints["symbols"])

    def test_inspect_python_tests_uses_targeted_pytest_files_and_imports(self) -> None:
        temp_root = self._fresh_dir("official-corpus-targeted-inspect")
        workspace = temp_root / "workspace"
        (workspace / "pkg").mkdir(parents=True, exist_ok=True)
        (workspace / "pkg" / "logic.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
        (workspace / "pkg" / "tests").mkdir(parents=True, exist_ok=True)
        (workspace / "pkg" / "tests" / "test_logic.py").write_text(
            "from pkg.logic import answer\n\n\ndef test_answer():\n    assert answer() == 2\n",
            encoding="utf-8",
        )

        summary = inspect_python_tests_with_context(
            workspace,
            target_test_files=["pkg/tests/test_logic.py"],
            prompt="Fix answer() in pkg.logic",
            meta={"FAIL_TO_PASS": json.dumps(["pkg/tests/test_logic.py::test_answer"])},
        )

        self.assertEqual(summary["targeted_test_files"], ["pkg/tests/test_logic.py"])
        self.assertIn("pkg/logic.py", summary["candidate_source_files"])
        self.assertIn("answer", summary["symbols"])

    def test_summarize_test_failures_tracks_missing_modules(self) -> None:
        failure = summarize_test_failures("ModuleNotFoundError: No module named 'erfa'")

        self.assertEqual(failure["missing_modules"], ["erfa"])
        self.assertEqual(_packages_for_missing_modules(failure["missing_modules"]), ["pyerfa"])

    def test_summarize_test_failures_detects_compiled_extension_bootstrap_needs(self) -> None:
        failure = summarize_test_failures(
            "ImportError: You appear to be trying to import astropy from within a source checkout "
            "without building the extension modules first.\n"
            "ModuleNotFoundError: No module named 'astropy.table._column_mixins'"
        )

        self.assertIn("compiled_extensions_missing", failure["environment_issues"])

    def test_packages_for_missing_modules_skips_internal_compiled_module_names(self) -> None:
        packages = _packages_for_missing_modules(["astropy.table._column_mixins", "erfa"])

        self.assertEqual(packages, ["pyerfa"])

    def test_workspace_repo_test_packages_reads_setup_cfg_test_dependencies(self) -> None:
        temp_root = self._fresh_dir("official-corpus-setupcfg-packages")
        workspace = temp_root / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "setup.cfg").write_text(
            "[options]\n"
            "install_requires =\n"
            "    packaging>=19.0\n"
            "    PyYAML>=3.13\n\n"
            "[options.extras_require]\n"
            "test =\n"
            "    pytest-astropy>=0.9\n"
            "    pytest-xdist\n",
            encoding="utf-8",
        )

        packages = _workspace_repo_test_packages(workspace, "astropy/astropy")

        self.assertIn("pytest-astropy>=0.9", packages)
        self.assertIn("pytest-xdist", packages)
        self.assertIn("packaging>=19.0", packages)

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
