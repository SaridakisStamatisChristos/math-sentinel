from __future__ import annotations

import unittest

from benchmarks.public_catalog import available_public_suite_groups, available_public_suites, load_public_suite


class PublicCatalogTests(unittest.TestCase):
    def test_available_public_suites_lists_expected_entries(self) -> None:
        self.assertEqual(
            set(available_public_suites()),
            {"swebench_verified_smoke", "swebench_verified_medium", "gaia_smoke", "gaia_medium", "math_public_smoke"},
        )

    def test_public_suite_groups_expose_smoke_and_medium_targets(self) -> None:
        groups = available_public_suite_groups()

        self.assertIn("public_smoke", groups)
        self.assertIn("public_medium", groups)
        self.assertIn("swebench_verified_medium", groups["public_medium"])
        self.assertIn("gaia_medium", groups["public_medium"])

    def test_load_public_suite_returns_swebench_suite(self) -> None:
        suite = load_public_suite("swebench_verified_smoke")

        self.assertEqual(suite.backend, "swebench_ops")
        self.assertEqual(len(suite.cases), 2)

    def test_load_public_suite_returns_gaia_suite(self) -> None:
        suite = load_public_suite("gaia_smoke")

        self.assertEqual(suite.backend, "gaia_ops")
        self.assertEqual(len(suite.cases), 3)

    def test_load_public_suite_returns_medium_swebench_suite(self) -> None:
        suite = load_public_suite("swebench_verified_medium")

        self.assertEqual(suite.backend, "swebench_ops")
        self.assertEqual(suite.tier, "medium")
        self.assertEqual(len(suite.cases), 4)
