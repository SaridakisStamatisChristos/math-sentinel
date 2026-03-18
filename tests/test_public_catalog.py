from __future__ import annotations

import unittest

from benchmarks.public_catalog import available_public_suites, load_public_suite


class PublicCatalogTests(unittest.TestCase):
    def test_available_public_suites_lists_expected_entries(self) -> None:
        self.assertEqual(
            set(available_public_suites()),
            {"swebench_verified_smoke", "gaia_smoke", "math_public_smoke"},
        )

    def test_load_public_suite_returns_swebench_suite(self) -> None:
        suite = load_public_suite("swebench_verified_smoke")

        self.assertEqual(suite.backend, "swebench_ops")
        self.assertEqual(len(suite.cases), 2)

    def test_load_public_suite_returns_gaia_suite(self) -> None:
        suite = load_public_suite("gaia_smoke")

        self.assertEqual(suite.backend, "gaia_ops")
        self.assertEqual(len(suite.cases), 3)
