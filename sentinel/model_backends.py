from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .model import TinyTransformerLM
from .tokenizer import StructuredTokenizer, build_default_tokenizer, get_structured_special_tokens

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency guard
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover - optional dependency guard
    BitsAndBytesConfig = None

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except Exception:  # pragma: no cover - optional dependency guard
    LoraConfig = None
    PeftModel = None
    get_peft_model = None


@dataclass
class ModelRuntimeInfo:
    provider: str
    backbone: str
    adapter_mode: str
    quantization: str
    dtype: str
    special_tokens_version: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "backbone": self.backbone,
            "adapter_mode": self.adapter_mode,
            "quantization": self.quantization,
            "dtype": self.dtype,
            "special_tokens_version": self.special_tokens_version,
        }


class HFTokenizerAdapter:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self._register_special_tokens()

    def _register_special_tokens(self) -> None:
        additional = [
            token
            for token in get_structured_special_tokens()
            if token not in {self.tokenizer.pad_token, self.tokenizer.bos_token, self.tokenizer.eos_token}
        ]
        special: Dict[str, Any] = {"additional_special_tokens": additional}
        if self.tokenizer.pad_token is None:
            special["pad_token"] = "<pad>"
        if self.tokenizer.bos_token is None:
            special["bos_token"] = "<bos>"
        if self.tokenizer.eos_token is None:
            special["eos_token"] = "<eos>"
        self.tokenizer.add_special_tokens(special)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def pad_id(self) -> int:
        return int(self.tokenizer.pad_token_id)

    @property
    def bos_id(self) -> int:
        return int(self.tokenizer.bos_token_id)

    @property
    def eos_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    def tokenize(self, text: str) -> List[str]:
        return list(self.tokenizer.tokenize(text))

    def token_to_id(self, token: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            return self.pad_id
        return int(token_id)

    def encode_unpadded(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )
        ids = list(encoded["input_ids"])
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode(self, text: str, seq_len: int) -> List[int]:
        ids = self.encode_unpadded(text, add_bos=True, add_eos=True)
        if len(ids) > seq_len:
            ids = ids[:seq_len]
            ids[-1] = self.eos_id
        ids += [self.pad_id] * (seq_len - len(ids))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        cleaned: List[int] = []
        for idx in ids:
            idx = int(idx)
            if idx == self.pad_id:
                continue
            cleaned.append(idx)
            if idx == self.eos_id:
                break
        return self.tokenizer.decode(cleaned, skip_special_tokens=False)


class LegacyTinyProver(TinyTransformerLM):
    provider_name = "legacy_tiny"


class HFCausalLMProver(nn.Module):
    provider_name = "hf_causal_lm"

    def __init__(self, model: Any, seq_len: int, pad_id: int, runtime_info: ModelRuntimeInfo) -> None:
        super().__init__()
        self.model = model
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.runtime_info = runtime_info

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=x, attention_mask=(x != self.pad_id).long())
        return out.logits

    @torch.no_grad()
    def generate_ids(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 24,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        generated = self.model.generate(
            input_ids=input_ids[:, -self.seq_len :],
            attention_mask=(input_ids[:, -self.seq_len :] != self.pad_id).long(),
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_k=top_k if top_k > 0 else None,
            eos_token_id=eos_id,
            pad_token_id=self.pad_id,
        )
        return generated


def _model_dtype(dtype_name: str) -> Optional[torch.dtype]:
    normalized = str(dtype_name or "float32").lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    return None


def _make_quantization_config(cfg: Dict[str, Any]) -> Any | None:
    quantization = str(cfg["model"].get("quantization", "none")).lower()
    if quantization == "none" or BitsAndBytesConfig is None:
        return None
    if quantization == "4bit":
        return BitsAndBytesConfig(load_in_4bit=True)
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def _hf_loader_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = cfg.get("model", {})
    kwargs: Dict[str, Any] = {}
    if bool(model_cfg.get("trust_remote_code", False)):
        kwargs["trust_remote_code"] = True
    if bool(model_cfg.get("local_files_only", False)):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        kwargs["local_files_only"] = True
    return kwargs


def _build_hf_tokenizer(backbone: str, cfg: Dict[str, Any]) -> HFTokenizerAdapter:
    if AutoTokenizer is None:
        raise RuntimeError("transformers is not installed")
    loader_kwargs = _hf_loader_kwargs(cfg)
    try:
        tokenizer = AutoTokenizer.from_pretrained(backbone, **loader_kwargs)
    except Exception:
        if not loader_kwargs.get("local_files_only"):
            loader_kwargs["local_files_only"] = True
            tokenizer = AutoTokenizer.from_pretrained(backbone, **loader_kwargs)
        else:
            raise
    return HFTokenizerAdapter(tokenizer)


def _maybe_apply_lora(model: Any, cfg: Dict[str, Any]) -> Any:
    adapter_mode = str(cfg["model"].get("adapter_mode", "none")).lower()
    if adapter_mode != "lora":
        return model
    if get_peft_model is None or LoraConfig is None:
        raise RuntimeError("peft is not installed")
    lora_cfg = LoraConfig(
        r=int(cfg["model"].get("lora_r", 8)),
        lora_alpha=int(cfg["model"].get("lora_alpha", 16)),
        lora_dropout=float(cfg["model"].get("lora_dropout", 0.05)),
        target_modules=cfg["model"].get("lora_target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)


def _build_hf_model(cfg: Dict[str, Any], tokenizer: HFTokenizerAdapter, device: str, for_training: bool) -> HFCausalLMProver:
    if AutoModelForCausalLM is None:
        raise RuntimeError("transformers is not installed")
    backbone = str(cfg["model"]["backbone"])
    model_kwargs: Dict[str, Any] = _hf_loader_kwargs(cfg)
    dtype = _model_dtype(str(cfg["model"].get("dtype", "float32")))
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    quantization_config = _make_quantization_config(cfg)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    device_map = str(cfg["model"].get("device_map", "auto")).strip()
    if device_map and device_map != "single":
        model_kwargs["device_map"] = device_map
    try:
        model = AutoModelForCausalLM.from_pretrained(backbone, **model_kwargs)
    except Exception:
        if not model_kwargs.get("local_files_only"):
            model_kwargs["local_files_only"] = True
            model = AutoModelForCausalLM.from_pretrained(backbone, **model_kwargs)
        else:
            raise
    model.resize_token_embeddings(tokenizer.vocab_size)
    if for_training:
        model = _maybe_apply_lora(model, cfg)
    runtime_info = ModelRuntimeInfo(
        provider="hf_causal_lm",
        backbone=backbone,
        adapter_mode=str(cfg["model"].get("adapter_mode", "none")),
        quantization=str(cfg["model"].get("quantization", "none")),
        dtype=str(cfg["model"].get("dtype", "float32")),
        special_tokens_version=int(cfg["model"].get("special_tokens_version", 2)),
    )
    prover = HFCausalLMProver(
        model=model,
        seq_len=int(cfg["model"]["seq_len"]),
        pad_id=tokenizer.pad_id,
        runtime_info=runtime_info,
    )
    return prover.to(device)


def ensure_model_config_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = cfg.setdefault("model", {})
    model_cfg.setdefault("provider", "legacy_tiny")
    model_cfg.setdefault("backbone", "Qwen/Qwen2.5-1.5B-Instruct")
    model_cfg.setdefault("adapter_mode", "none")
    model_cfg.setdefault("quantization", "none")
    model_cfg.setdefault("dtype", "float32")
    model_cfg.setdefault("device_map", "single")
    model_cfg.setdefault("trust_remote_code", False)
    model_cfg.setdefault("local_files_only", False)
    model_cfg.setdefault("special_tokens_version", 2)
    model_cfg.setdefault("lora_r", 8)
    model_cfg.setdefault("lora_alpha", 16)
    model_cfg.setdefault("lora_dropout", 0.05)
    model_cfg.setdefault("lora_target_modules", "all-linear")
    return cfg


def build_runtime_tokenizer(cfg: Dict[str, Any]) -> Any:
    ensure_model_config_defaults(cfg)
    provider = str(cfg["model"].get("provider", "legacy_tiny")).lower()
    if provider in {"hf", "hf_causal_lm", "openweight"}:
        return _build_hf_tokenizer(str(cfg["model"]["backbone"]), cfg)
    return build_default_tokenizer()


def build_prover(
    cfg: Dict[str, Any],
    tokenizer: Any,
    device: str,
    *,
    for_training: bool = False,
) -> nn.Module:
    ensure_model_config_defaults(cfg)
    provider = str(cfg["model"].get("provider", "legacy_tiny")).lower()
    if provider in {"hf", "hf_causal_lm", "openweight"}:
        if not isinstance(tokenizer, HFTokenizerAdapter):
            raise TypeError("HF provider requires HFTokenizerAdapter")
        return _build_hf_model(cfg, tokenizer, device, for_training)
    return LegacyTinyProver(
        vocab_size=tokenizer.vocab_size,
        seq_len=int(cfg["model"]["seq_len"]),
        d_model=int(cfg["model"]["d_model"]),
        n_heads=int(cfg["model"]["n_heads"]),
        n_layers=int(cfg["model"]["n_layers"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)


def create_prover_and_tokenizer(cfg: Dict[str, Any], device: str, *, for_training: bool = False) -> Tuple[nn.Module, Any]:
    tokenizer = build_runtime_tokenizer(cfg)
    prover = build_prover(cfg, tokenizer, device, for_training=for_training)
    return prover, tokenizer


def build_model_runtime_info(cfg: Dict[str, Any], prover: nn.Module) -> Dict[str, Any]:
    ensure_model_config_defaults(cfg)
    if isinstance(prover, HFCausalLMProver):
        return prover.runtime_info.to_dict()
    return ModelRuntimeInfo(
        provider="legacy_tiny",
        backbone="legacy_tiny",
        adapter_mode=str(cfg["model"].get("adapter_mode", "none")),
        quantization=str(cfg["model"].get("quantization", "none")),
        dtype=str(cfg["model"].get("dtype", "float32")),
        special_tokens_version=int(cfg["model"].get("special_tokens_version", 2)),
    ).to_dict()
