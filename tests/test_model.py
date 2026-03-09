from __future__ import annotations

import unittest

import torch

from sentinel.model import CausalSelfAttention


class ModelTests(unittest.TestCase):
    def test_attention_mask_fill_value_is_representable_in_float16(self) -> None:
        att = torch.zeros((1, 1, 2, 2), dtype=torch.float16)

        fill_value = CausalSelfAttention._mask_fill_value(att)

        self.assertEqual(fill_value, torch.finfo(torch.float16).min)
        masked = att.masked_fill(torch.tensor([[[[False, True], [False, False]]]]), fill_value)
        self.assertTrue(torch.isfinite(masked).all())


if __name__ == "__main__":
    unittest.main()