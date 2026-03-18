from __future__ import annotations

import unittest

from search.transposition import TranspositionTable


class TranspositionTests(unittest.TestCase):
    def test_register_accepts_new_then_rejects_non_improving_duplicate(self) -> None:
        table = TranspositionTable(capacity=4)
        accepted, novelty = table.register("state_a", 0.5, 1)
        self.assertTrue(accepted)
        self.assertEqual(novelty, 1.0)

        accepted, novelty = table.register("state_a", 0.4, 2)
        self.assertFalse(accepted)
        self.assertLess(novelty, 1.0)

        accepted, novelty = table.register("state_a", 0.7, 1)
        self.assertTrue(accepted)
        self.assertLessEqual(novelty, 0.5)
