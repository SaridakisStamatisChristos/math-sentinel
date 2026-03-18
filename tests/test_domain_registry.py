from __future__ import annotations

import unittest

from domains import available_backends, create_reasoning_domain, default_curriculum_config
from domains.code_ops.backend import CodeOpsReasoningDomain
from domains.gaia_ops.backend import GaiaOpsReasoningDomain
from domains.math.backend import MathReasoningDomain
from domains.planning_ops.backend import PlanningOpsReasoningDomain
from domains.string_ops.backend import StringOpsReasoningDomain
from domains.swebench_ops.backend import SwebenchOpsReasoningDomain


class DomainRegistryTests(unittest.TestCase):
    def test_create_reasoning_domain_returns_math_backend(self) -> None:
        backend = create_reasoning_domain("math")
        self.assertIsInstance(backend, MathReasoningDomain)
        self.assertEqual(default_curriculum_config("math"), "config/curriculum.yaml")

    def test_create_reasoning_domain_accepts_string_aliases(self) -> None:
        backend = create_reasoning_domain("strings")
        self.assertIsInstance(backend, StringOpsReasoningDomain)
        self.assertEqual(default_curriculum_config("text_ops"), "config/string_ops_curriculum.yaml")

    def test_create_reasoning_domain_accepts_code_aliases(self) -> None:
        backend = create_reasoning_domain("code")
        self.assertIsInstance(backend, CodeOpsReasoningDomain)
        self.assertEqual(default_curriculum_config("code_ops"), "config/code_ops_curriculum.yaml")

    def test_available_backends_lists_all_registered_domains(self) -> None:
        self.assertEqual(set(available_backends()), {"math", "string_ops", "code_ops", "planning_ops", "swebench_ops", "gaia_ops"})

    def test_create_reasoning_domain_accepts_planning_aliases(self) -> None:
        backend = create_reasoning_domain("planning")
        self.assertIsInstance(backend, PlanningOpsReasoningDomain)
        self.assertEqual(default_curriculum_config("planner"), "config/planning_ops_curriculum.yaml")

    def test_create_reasoning_domain_accepts_swebench_aliases(self) -> None:
        backend = create_reasoning_domain("swebench")
        self.assertIsInstance(backend, SwebenchOpsReasoningDomain)
        self.assertEqual(default_curriculum_config("swe_bench"), "config/swebench_ops_curriculum.yaml")

    def test_create_reasoning_domain_passes_runtime_config_to_benchmark_backends(self) -> None:
        backend = create_reasoning_domain("gaia", runtime_config={"benchmark": {"assistance_mode": "assisted", "oracle_hints_enabled": True}})

        self.assertIsInstance(backend, GaiaOpsReasoningDomain)
        self.assertEqual(backend.assistance_mode, "assisted")
        self.assertTrue(backend.oracle_hints_enabled)

    def test_create_reasoning_domain_accepts_gaia_aliases(self) -> None:
        backend = create_reasoning_domain("gaia")
        self.assertIsInstance(backend, GaiaOpsReasoningDomain)
        self.assertEqual(default_curriculum_config("gaia_ops"), "config/gaia_ops_curriculum.yaml")


if __name__ == "__main__":
    unittest.main()
