import unittest

from text_utils import slugify


class SlugifyTests(unittest.TestCase):
    def test_slugify_uses_hyphen_separator(self):
        self.assertEqual(slugify("Ada Lovelace"), "ada-lovelace")


if __name__ == "__main__":
    unittest.main()
