import unittest

from filters import is_positive


class PositiveFilterTests(unittest.TestCase):
    def test_zero_is_not_positive(self):
        self.assertFalse(is_positive(0))

    def test_positive_numbers_still_pass(self):
        self.assertTrue(is_positive(7))


if __name__ == "__main__":
    unittest.main()
