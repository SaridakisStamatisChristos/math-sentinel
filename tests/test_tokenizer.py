from __future__ import annotations

import unittest

from sentinel.tokenizer import build_default_tokenizer


class StructuredTokenizerTests(unittest.TestCase):
    def test_tokenizer_uses_structured_markers(self) -> None:
        tokenizer = build_default_tokenizer()
        text = '[DOMAIN] math\nACTION {"type":"ANSWER","content":"5"}'

        tokens = tokenizer.tokenize(text)

        self.assertIn("[DOMAIN]", tokens)
        self.assertIn("math", tokens)
        self.assertIn("\n", tokens)
        self.assertIn("ACTION ", tokens)
        self.assertIn("ANSWER", tokens)

    def test_encode_decode_round_trip_preserves_canonical_text(self) -> None:
        tokenizer = build_default_tokenizer()
        text = 'ACTION {"type":"APPLY","tool":"add","content":"2,3"}'

        decoded = tokenizer.decode(tokenizer.encode(text, seq_len=256))

        self.assertEqual(decoded, text)

    def test_domain_specific_symbols_are_tokenized_as_structured_units(self) -> None:
        tokenizer = build_default_tokenizer()
        text = "planning_ops day_plan has_conditional assignment_count"

        tokens = tokenizer.tokenize(text)

        self.assertEqual(tokens, ["planning_ops", " ", "day_plan", " ", "has_conditional", " ", "assignment_count"])


if __name__ == "__main__":
    unittest.main()
