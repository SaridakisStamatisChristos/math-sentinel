import unittest

from formatters import format_name


class FormatterTests(unittest.TestCase):
    def test_format_name_preserves_argument_order(self):
        self.assertEqual(format_name("Ada", "Lovelace"), "Ada Lovelace")


if __name__ == "__main__":
    unittest.main()
