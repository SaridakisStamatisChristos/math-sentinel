
from __future__ import annotations

from typing import List

import torch

from .model import TinyTransformerLM
from .tokenizer import StructuredTokenizer


def generate_text(
    model: TinyTransformerLM,
    tokenizer: StructuredTokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 24,
) -> str:
    ids = tokenizer.encode(prompt, model.seq_len)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    valid = (x[0] != tokenizer.pad_id).nonzero(as_tuple=False)
    last = int(valid[-1].item()) + 1 if len(valid) else 1
    x = x[:, :last]
    out = model.generate_ids(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
    return tokenizer.decode(out[0].tolist())


def propose_actions(
    model: TinyTransformerLM,
    tokenizer: StructuredTokenizer,
    prompt: str,
    device: str,
    proposal_count: int = 4,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 24,
) -> List[str]:
    proposals = []
    for _ in range(proposal_count):
        proposals.append(
            generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        )
    return proposals
