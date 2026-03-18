import unittest

from app import positive_only


class FilterBugTests(unittest.TestCase):
    def test_positive_only_excludes_zero(self):
        self.assertEqual(positive_only([-2, 0, 1, 3]), [1, 3])


if __name__ == "__main__":
    unittest.main()
