
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]

STATE_MARKERS = [
    "[DOMAIN]", "[PROBLEM]", "[GOAL]", "[ASSUMPTIONS]", "[DERIVED]",
    "[SUBGOALS]", "[LEMMAS]", "[TOOL_HISTORY]", "[ACTION_HISTORY]",
    "[STATUS]", "[FINAL_ANSWER]", "[METADATA]", "[END_STATE]",
]

ACTION_MARKERS = [
    '<action type="', '"></action>', '</action>', ' tool="', ' name="',
    '<result>', '</result>', '<note>', '</note>', '<answer>', '</answer>',
]

ASCII_CHARS = [chr(i) for i in range(32, 127)] + ["\n", "\t"]


@dataclass
class CharTokenizer:
    vocab: List[str]

    def __post_init__(self) -> None:
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.vocab)}
        self.itos: Dict[int, str] = {i: tok for tok, i in self.stoi.items()}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, seq_len: int) -> List[int]:
        ids = [self.bos_id]
        for ch in text:
            ids.append(self.stoi.get(ch, self.stoi.get("?", self.pad_id)))
        ids.append(self.eos_id)
        if len(ids) > seq_len:
            ids = ids[:seq_len]
            ids[-1] = self.eos_id
        ids += [self.pad_id] * (seq_len - len(ids))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        out: List[str] = []
        for idx in ids:
            idx = int(idx)
            if idx in (self.pad_id, self.bos_id):
                continue
            if idx == self.eos_id:
                break
            out.append(self.itos.get(idx, "?"))
        return "".join(out)


def build_default_tokenizer() -> CharTokenizer:
    vocab = SPECIAL_TOKENS + ASCII_CHARS
    return CharTokenizer(vocab=vocab)
