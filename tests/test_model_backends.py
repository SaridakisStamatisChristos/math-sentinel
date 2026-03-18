from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from sentinel.model_backends import (
    HFTokenizerAdapter,
    HFCausalLMProver,
    build_model_runtime_info,
    create_prover_and_tokenizer,
    ensure_model_config_defaults,
)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token = None
        self.bos_token = None
        self.eos_token = None
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 99
        self.added_special_tokens: list[str] = []

    def add_special_tokens(self, payload: dict[str, object]) -> int:
        self.pad_token = str(payload.get("pad_token", self.pad_token or "<pad>"))
        self.bos_token = str(payload.get("bos_token", self.bos_token or "<bos>"))
        self.eos_token = str(payload.get("eos_token", self.eos_token or "<eos>"))
        self.added_special_tokens.extend(list(payload.get("additional_special_tokens", [])))
        return len(self.added_special_tokens)

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def convert_tokens_to_ids(self, token: str) -> int:
        return len(token)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [len(part) for part in text.split()]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return " ".join(str(item) for item in ids)

    def __len__(self) -> int:
        return 128 + len(self.added_special_tokens)


class _FakeHFModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(256, 16)
        self.head = torch.nn.Linear(16, 256)
        self.resized_to = 0

    def resize_token_embeddings(self, size: int) -> None:
        self.resized_to = size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> object:
        logits = self.head(self.emb(input_ids))
        return type("Output", (), {"logits": logits})

    def generate(self, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        next_token = torch.full((input_ids.shape[0], 1), 7, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, next_token], dim=1)


class ModelBackendTests(unittest.TestCase):
    def test_config_defaults_are_added_for_model_stack(self) -> None:
        cfg = ensure_model_config_defaults({"model": {"seq_len": 32, "d_model": 16, "n_heads": 2, "n_layers": 1, "dropout": 0.1}})

        self.assertEqual(cfg["model"]["provider"], "legacy_tiny")
        self.assertEqual(cfg["model"]["backbone"], "Qwen/Qwen2.5-1.5B-Instruct")

    def test_hf_tokenizer_adapter_registers_structured_special_tokens(self) -> None:
        adapter = HFTokenizerAdapter(_FakeTokenizer())

        self.assertGreater(adapter.vocab_size, 128)
        self.assertIn("[DOMAIN]", adapter.tokenizer.added_special_tokens)
        self.assertEqual(adapter.pad_id, 0)

    @patch("sentinel.model_backends._maybe_apply_lora", side_effect=lambda model, cfg: model)
    @patch("sentinel.model_backends.AutoModelForCausalLM")
    @patch("sentinel.model_backends.AutoTokenizer")
    def test_create_prover_and_tokenizer_supports_hf_provider(self, mock_tokenizer_cls: object, mock_model_cls: object, _: object) -> None:
        mock_tokenizer_cls.from_pretrained.return_value = _FakeTokenizer()
        mock_model_cls.from_pretrained.return_value = _FakeHFModel()
        cfg = ensure_model_config_defaults(
            {
                "model": {
                    "provider": "hf_causal_lm",
                    "backbone": "fake/model",
                    "adapter_mode": "none",
                    "seq_len": 32,
                    "d_model": 16,
                    "n_heads": 2,
                    "n_layers": 1,
                    "dropout": 0.1,
                }
            }
        )

        prover, tokenizer = create_prover_and_tokenizer(cfg, "cpu", for_training=False)

        self.assertIsInstance(prover, HFCausalLMProver)
        self.assertEqual(tokenizer.pad_id, 0)
        self.assertEqual(build_model_runtime_info(cfg, prover)["provider"], "hf_causal_lm")


if __name__ == "__main__":
    unittest.main()
