from __future__ import annotations

import unittest

from benchmarks.ablations import available_benchmark_ablations, resolve_benchmark_ablations
from benchmarks.profiles import apply_benchmark_profile, available_benchmark_profiles, resolve_benchmark_profiles
from benchmark_v7 import default_campaign_profile_for_suite
from sentinel.config import load_runtime_config


class BenchmarkProfileTests(unittest.TestCase):
    def test_profile_catalog_lists_expected_profiles(self) -> None:
        profiles = set(available_benchmark_profiles())

        self.assertIn("smoke_tiny", profiles)
        self.assertIn("public_claim_no_repairs", profiles)
        self.assertIn("public_claim_blind_structural", profiles)
        self.assertIn("public_claim_coder_local_1p5b", profiles)
        self.assertIn("public_unassisted_strict", profiles)
        self.assertIn("qwen_coder_flagship_32b", profiles)
        self.assertIn("rtx4060_general_local", profiles)
        self.assertIn("rtx4060_coder_local", profiles)

    def test_profile_config_extends_public_smoke_snapshot(self) -> None:
        cfg = load_runtime_config("config/benchmarks/profile_qwen_coder_32b.yaml", search_config_path="")

        self.assertEqual(cfg["model"]["provider"], "hf_causal_lm")
        self.assertEqual(cfg["model"]["backbone"], "Qwen/Qwen2.5-Coder-32B-Instruct")
        self.assertEqual(cfg["search"]["beam_width"], 10)
        self.assertTrue(cfg["runtime"]["safe_mode"])
        self.assertEqual(cfg["benchmark"]["assistance_mode"], "unassisted")

    def test_rtx4060_profile_is_tuned_for_single_gpu_1p5b_runtime(self) -> None:
        cfg = load_runtime_config("config/benchmarks/profile_rtx4060_general_1p5b.yaml", search_config_path="")

        self.assertEqual(cfg["model"]["provider"], "hf_causal_lm")
        self.assertEqual(cfg["model"]["backbone"], "models/Qwen2.5-1.5B-Instruct")
        self.assertEqual(cfg["model"]["device_map"], "single")
        self.assertEqual(cfg["model"]["dtype"], "float16")
        self.assertTrue(cfg["model"]["local_files_only"])
        self.assertEqual(cfg["search"]["beam_width"], 5)
        self.assertEqual(cfg["training"]["micro_batch_size"], 1)

    def test_rtx4060_coder_profile_allows_full_repo_repair_horizon(self) -> None:
        cfg = load_runtime_config("config/benchmarks/profile_rtx4060_coder_1p5b.yaml", search_config_path="")

        self.assertEqual(cfg["model"]["provider"], "hf_causal_lm")
        self.assertEqual(cfg["model"]["backbone"], "models/Qwen2.5-Coder-1.5B-Instruct")
        self.assertTrue(cfg["model"]["local_files_only"])
        self.assertEqual(cfg["search"]["mode"], "beam")
        self.assertEqual(cfg["search"]["beam_width"], 8)
        self.assertEqual(cfg["search"]["proposal_count"], 8)
        self.assertEqual(cfg["search"]["max_depth"], 8)
        self.assertEqual(cfg["search"]["mcts_simulations"], 24)
        self.assertEqual(cfg["search"]["transposition_capacity"], 8192)
        self.assertEqual(cfg["search"]["fallback_bonus"], 0.35)

    def test_rtx4060_product_config_preserves_local_search_when_override_is_disabled(self) -> None:
        cfg = load_runtime_config("config/product_rtx4060_laptop.yaml", search_config_path="")

        self.assertEqual(cfg["model"]["provider"], "hf_causal_lm")
        self.assertEqual(cfg["model"]["backbone"], "models/Qwen2.5-1.5B-Instruct")
        self.assertEqual(cfg["model"]["dtype"], "float16")
        self.assertEqual(cfg["model"]["device_map"], "single")
        self.assertTrue(cfg["model"]["local_files_only"])
        self.assertFalse(cfg["runtime"]["deterministic"])
        self.assertEqual(cfg["search"]["beam_width"], 5)
        self.assertEqual(cfg["training"]["micro_batch_size"], 1)

    def test_apply_benchmark_profile_uses_catalog_config_snapshot(self) -> None:
        base_cfg = load_runtime_config("config/default.yaml", search_config_path="")
        profile = resolve_benchmark_profiles("smoke_tiny")[0]

        cfg = apply_benchmark_profile(base_cfg, profile)

        self.assertEqual(cfg["model"]["provider"], "legacy_tiny")
        self.assertEqual(cfg["search"]["beam_width"], 4)
        self.assertTrue(cfg["runtime"]["safe_mode"])
        self.assertEqual(cfg["benchmark"]["assistance_mode"], "unassisted")
        self.assertFalse(cfg["search"]["guided_fallback_rollout"])

    def test_default_public_campaign_profile_uses_no_repairs_claim_lane(self) -> None:
        self.assertEqual(default_campaign_profile_for_suite("public_medium"), "public_claim_no_repairs")
        self.assertEqual(default_campaign_profile_for_suite("swebench_verified_medium"), "public_claim_coder_local_1p5b")
        self.assertEqual(default_campaign_profile_for_suite("official:swebench"), "public_claim_coder_local_1p5b")
        self.assertEqual(default_campaign_profile_for_suite("official:gaia"), "public_claim_no_repairs")
        self.assertEqual(
            default_campaign_profile_for_suite("manifest:benchmarks/manifests/swebench_verified_medium_official_style.json"),
            "public_claim_coder_local_1p5b",
        )
        self.assertEqual(
            default_campaign_profile_for_suite("manifest:benchmarks/manifests/gaia_medium_official_style.json"),
            "public_claim_no_repairs",
        )

    def test_strict_and_search_assisted_public_profiles_diverge_on_guided_rollout(self) -> None:
        blind_cfg = load_runtime_config("config/benchmarks/profile_public_claim_blind_structural.yaml", search_config_path="")
        coder_claim_cfg = load_runtime_config("config/benchmarks/profile_public_claim_coder_local_1p5b.yaml", search_config_path="")
        claim_cfg = load_runtime_config("config/benchmarks/profile_public_claim_no_repairs.yaml", search_config_path="")
        strict_cfg = load_runtime_config("config/benchmarks/profile_public_unassisted_strict.yaml", search_config_path="")
        assisted_cfg = load_runtime_config("config/benchmarks/profile_public_search_assisted.yaml", search_config_path="")

        self.assertEqual(claim_cfg["model"]["provider"], "hf_causal_lm")
        self.assertEqual(claim_cfg["model"]["backbone"], "models/Qwen2.5-1.5B-Instruct")
        self.assertTrue(claim_cfg["model"]["local_files_only"])
        self.assertEqual(claim_cfg["model"]["dtype"], "float16")
        self.assertEqual(claim_cfg["benchmark"]["report_lane"], "claim_no_repairs")
        self.assertFalse(claim_cfg["search"]["enable_fallback_repairs"])
        self.assertFalse(claim_cfg["search"]["guided_fallback_rollout"])
        self.assertEqual(claim_cfg["memory"]["retrieval_mode"], "none")
        self.assertEqual(blind_cfg["benchmark"]["report_lane"], "claim_blind_structural")
        self.assertTrue(blind_cfg["benchmark"]["blind_structural_mode"])
        self.assertFalse(blind_cfg["benchmark"]["allow_named_family_routing"])
        self.assertFalse(blind_cfg["benchmark"]["allow_errata_overrides"])
        self.assertTrue(blind_cfg["search"]["prompt_compaction"])
        self.assertEqual(blind_cfg["search"]["prompt_problem_chars"], 420)
        self.assertEqual(blind_cfg["search"]["prompt_tool_limit"], 1)
        self.assertEqual(coder_claim_cfg["model"]["backbone"], "models/Qwen2.5-Coder-1.5B-Instruct")
        self.assertEqual(coder_claim_cfg["benchmark"]["report_lane"], "claim_no_repairs_coder")
        self.assertEqual(coder_claim_cfg["search"]["beam_width"], 8)
        self.assertEqual(strict_cfg["benchmark"]["assistance_mode"], "unassisted")
        self.assertTrue(strict_cfg["benchmark"]["claim_mode"])
        self.assertFalse(strict_cfg["search"]["guided_fallback_rollout"])
        self.assertFalse(strict_cfg["search"]["deterministic_fallback_chain"])
        self.assertFalse(strict_cfg["search"]["enable_fallback_repairs"])
        self.assertTrue(strict_cfg["benchmark"]["fail_on_integrity_violation"])
        self.assertEqual(strict_cfg["memory"]["retrieval_mode"], "none")
        self.assertFalse(assisted_cfg["benchmark"]["claim_mode"])
        self.assertTrue(assisted_cfg["search"]["guided_fallback_rollout"])
        self.assertTrue(assisted_cfg["search"]["enable_fallback_repairs"])

    def test_ablation_catalog_lists_expected_entries(self) -> None:
        ablations = set(available_benchmark_ablations())

        self.assertIn("baseline", ablations)
        self.assertIn("no_retrieval", ablations)

    def test_resolve_benchmark_ablations_preserves_order(self) -> None:
        ablations = resolve_benchmark_ablations("baseline,no_guided_rollout")

        self.assertEqual([ablation.name for ablation in ablations], ["baseline", "no_guided_rollout"])
