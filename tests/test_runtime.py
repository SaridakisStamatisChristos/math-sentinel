from __future__ import annotations

import unittest

from sentinel.runtime import apply_safe_runtime_profile, configure_runtime, ensure_runtime_config_defaults


class RuntimeTests(unittest.TestCase):
    def test_safe_runtime_profile_forces_strict_decoder(self) -> None:
        cfg = {
            "seed": 1,
            "device": "cpu",
            "search": {"decoder_mode": "hybrid", "temperature": 0.8, "top_k": 20},
            "runtime": {"safe_mode": True, "deterministic": True},
        }
        ensure_runtime_config_defaults(cfg)
        apply_safe_runtime_profile(cfg)
        self.assertEqual(cfg["search"]["decoder_mode"], "strict")
        self.assertEqual(cfg["search"]["temperature"], 0.0)
        self.assertEqual(cfg["search"]["top_k"], 1)

    def test_configure_runtime_returns_device(self) -> None:
        cfg = {"seed": 1, "device": "cpu", "search": {}, "runtime": {}}
        device = configure_runtime(cfg, deterministic_override=True, safe_override=True)
        self.assertEqual(device, "cpu")
        self.assertTrue(cfg["runtime"]["deterministic"])
        self.assertTrue(cfg["runtime"]["safe_mode"])
