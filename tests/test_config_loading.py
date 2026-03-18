from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from sentinel.config import load_runtime_config


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"


class ConfigLoadingTests(unittest.TestCase):
    def _fresh_dir(self, name: str) -> Path:
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        path = TEST_TMP_ROOT / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def test_runtime_config_merges_search_file_into_search_section(self) -> None:
        root = self._fresh_dir("config-loading")
        config_path = root / "default.yaml"
        search_path = root / "search.yaml"

        config_path.write_text(
            "search:\n"
            "  beam_width: 8\n"
            "  max_depth: 6\n"
            "  temperature: 0.8\n"
            "training:\n"
            "  max_new_tokens: 64\n",
            encoding="utf-8",
        )
        search_path.write_text(
            "beam_width: 3\n"
            "goal_bonus: 0.9\n",
            encoding="utf-8",
        )

        cfg = load_runtime_config(str(config_path), str(search_path))

        self.assertEqual(cfg["search"]["beam_width"], 3)
        self.assertEqual(cfg["search"]["max_depth"], 6)
        self.assertEqual(cfg["search"]["goal_bonus"], 0.9)
        self.assertEqual(cfg["training"]["max_new_tokens"], 64)
        self.assertEqual(cfg["model"]["provider"], "legacy_tiny")
        self.assertEqual(cfg["model"]["backbone"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertFalse(cfg["runtime"]["safe_mode"])
        self.assertTrue(cfg["runtime"]["structured_logs"])
        self.assertEqual(cfg["benchmark"]["assistance_mode"], "unassisted")

    def test_runtime_config_supports_extends_without_search_overlay(self) -> None:
        root = self._fresh_dir("config-extends")
        base_path = root / "base.yaml"
        child_path = root / "child.yaml"
        search_path = root / "search.yaml"

        base_path.write_text(
            "runtime:\n"
            "  safe_mode: true\n"
            "search:\n"
            "  beam_width: 6\n"
            "model:\n"
            "  provider: legacy_tiny\n"
            "  seq_len: 32\n"
            "  d_model: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 1\n"
            "  dropout: 0.1\n",
            encoding="utf-8",
        )
        child_path.write_text(
            "extends: base.yaml\n"
            "search:\n"
            "  beam_width: 9\n"
            "model:\n"
            "  backbone: fake/backbone\n",
            encoding="utf-8",
        )
        search_path.write_text("beam_width: 2\n", encoding="utf-8")

        cfg = load_runtime_config(str(child_path), search_config_path="")

        self.assertEqual(cfg["search"]["beam_width"], 9)
        self.assertTrue(cfg["runtime"]["safe_mode"])
        self.assertEqual(cfg["model"]["backbone"], "fake/backbone")
        self.assertEqual(cfg["benchmark"]["assistance_mode"], "unassisted")


if __name__ == "__main__":
    unittest.main()
