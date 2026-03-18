import unittest

from stats import count_items


class CountBugTests(unittest.TestCase):
    def test_count_items_returns_full_length(self):
        self.assertEqual(count_items(["alpha", "beta", "gamma"]), 3)


if __name__ == "__main__":
    unittest.main()
