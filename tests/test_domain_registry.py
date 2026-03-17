from __future__ import annotations

import unittest

from domains import create_reasoning_domain, default_curriculum_config
from domains.math.backend import MathReasoningDomain
from domains.string_ops.backend import StringOpsReasoningDomain


class DomainRegistryTests(unittest.TestCase):
    def test_create_reasoning_domain_returns_math_backend(self) -> None:
        backend = create_reasoning_domain("math")
        self.assertIsInstance(backend, MathReasoningDomain)
        self.assertEqual(default_curriculum_config("math"), "config/curriculum.yaml")

    def test_create_reasoning_domain_accepts_string_aliases(self) -> None:
        backend = create_reasoning_domain("strings")
        self.assertIsInstance(backend, StringOpsReasoningDomain)
        self.assertEqual(default_curriculum_config("text_ops"), "config/string_ops_curriculum.yaml")


if __name__ == "__main__":
    unittest.main()
