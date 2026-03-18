import unittest

from text_utils import slugify


class SlugifyBugTests(unittest.TestCase):
    def test_slugify_collapses_spaces_and_uses_hyphen(self):
        self.assertEqual(slugify("Ada   Lovelace"), "ada-lovelace")


if __name__ == "__main__":
    unittest.main()
