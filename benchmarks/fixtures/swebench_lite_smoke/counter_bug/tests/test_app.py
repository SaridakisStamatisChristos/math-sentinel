import unittest

from app import add


class CounterBugTests(unittest.TestCase):
    def test_add_returns_sum(self):
        self.assertEqual(add(2, 3), 5)


if __name__ == "__main__":
    unittest.main()
