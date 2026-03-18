from __future__ import annotations

import unittest

from benchmarks.ablations import available_benchmark_ablations, resolve_benchmark_ablations
from benchmarks.profiles import apply_benchmark_profile, available_benchmark_profiles, resolve_benchmark_profiles
from sentinel.config import load_runtime_config


class BenchmarkProfileTests(unittest.TestCase):
    def test_profile_catalog_lists_expected_profiles(self) -> None:
        profiles = set(available_benchmark_profiles())

        self.assertIn("smoke_tiny", profiles)
        self.assertIn("qwen_coder_flagship_32b", profiles)

    def test_profile_config_extends_public_smoke_snapshot(self) -> None:
        cfg = load_runtime_config("config/benchmarks/profile_qwen_coder_32b.yaml", search_config_path="")

        self.assertEqual(cfg["model"]["provider"], "hf_causal_lm")
        self.assertEqual(cfg["model"]["backbone"], "Qwen/Qwen2.5-Coder-32B-Instruct")
        self.assertEqual(cfg["search"]["beam_width"], 10)
        self.assertTrue(cfg["runtime"]["safe_mode"])

    def test_apply_benchmark_profile_uses_catalog_config_snapshot(self) -> None:
        base_cfg = load_runtime_config("config/default.yaml", search_config_path="")
        profile = resolve_benchmark_profiles("smoke_tiny")[0]

        cfg = apply_benchmark_profile(base_cfg, profile)

        self.assertEqual(cfg["model"]["provider"], "legacy_tiny")
        self.assertEqual(cfg["search"]["beam_width"], 4)
        self.assertTrue(cfg["runtime"]["safe_mode"])

    def test_ablation_catalog_lists_expected_entries(self) -> None:
        ablations = set(available_benchmark_ablations())

        self.assertIn("baseline", ablations)
        self.assertIn("no_retrieval", ablations)

    def test_resolve_benchmark_ablations_preserves_order(self) -> None:
        ablations = resolve_benchmark_ablations("baseline,no_guided_rollout")

        self.assertEqual([ablation.name for ablation in ablations], ["baseline", "no_guided_rollout"])
