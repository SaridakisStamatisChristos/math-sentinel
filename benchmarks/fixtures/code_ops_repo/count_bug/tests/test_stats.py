import unittest

from stats import item_count


class CountBugTests(unittest.TestCase):
    def test_item_count_returns_full_length(self):
        self.assertEqual(item_count(["a", "b", "c"]), 3)


if __name__ == "__main__":
    unittest.main()
