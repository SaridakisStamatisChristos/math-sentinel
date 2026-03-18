from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]

STATE_MARKERS = [
    "[DOMAIN]", "[PROBLEM]", "[GOAL]", "[ASSUMPTIONS]", "[DERIVED]",
    "[SUBGOALS]", "[LEMMAS]", "[TOOL_HISTORY]", "[ACTION_HISTORY]",
    "[STATUS]", "[FINAL_ANSWER]", "[METADATA]", "[END_STATE]",
    "[RETRIEVED_LEMMAS]", "[SIMILAR_HARD_CASES]", "[TACTIC_HINTS]",
    "[PROVENANCE]", "[EVIDENCE]", "[OBLIGATIONS]", "[DEPENDENCIES]",
    "[TOOL_PAYLOADS]", "[TERMINAL_CONFIDENCE]", "[RETRIEVAL_TOOL_HINTS]",
    "[FAILURE_AVOIDANCE]",
]

LEGACY_ACTION_MARKERS = [
    '<action type="', '"></action>', '</action>', ' tool="', ' name="',
    '<result>', '</result>', '<note>', '</note>', '<answer>', '</answer>',
]

CANONICAL_ACTION_TOKENS = [
    "ACTION ", '"type"', '"content"', '"tool"', '"name"', '"payload"',
    '{"type":"', '","content":"', '","tool":"', '","name":"', '"}',
]

ACTION_TYPES = [
    "THINK", "APPLY", "CHECK", "ANSWER", "REWRITE", "LEMMA", "SUBGOAL",
    "RESOLVE_SUBGOAL", "ASSUME", "BACKTRACK", "CALL_PLUGIN", "SIMPLIFY",
]

DOMAIN_TOKENS = [
    "math", "string_ops", "code_ops", "planning_ops", "swebench_ops", "gaia_ops",
    "arithmetic", "fractions", "divmod", "gcd_lcm", "modular", "primality", "factorization",
    "linear_equation", "polynomial_simplify", "derivative", "integral", "parity_proof",
    "reverse_text", "uppercase_text", "vowel_count", "sort_words", "dedupe_words",
    "function_name", "parameter_count", "has_loop", "first_called_function", "return_literal",
    "has_conditional", "assignment_count", "called_function_count", "repo_patch",
    "project_plan", "shopping_plan", "day_plan",
    "swebench_patch", "gaia_csv_reasoning", "gaia_json_reasoning", "gaia_schedule_reasoning",
]

CODE_TOKENS = [
    "def", "return", "for", "while", "if", "else", "in", "range", "print",
    "True", "False", "None", "yes", "no",
]

PLANNING_TOKENS = [
    "shopping_plan", "project_plan", "day_plan",
    "step_1", "step_2", "step_3", "duration", "budget", "priority",
    "deps", "time_limit",
]

WHITESPACE_TOKENS = ["\n\n", "\n", "    ", "  ", "\t"]

ASCII_CHARS = [chr(i) for i in range(32, 127)]


def _build_vocab() -> List[str]:
    vocab = (
        SPECIAL_TOKENS
        + STATE_MARKERS
        + LEGACY_ACTION_MARKERS
        + CANONICAL_ACTION_TOKENS
        + ACTION_TYPES
        + DOMAIN_TOKENS
        + CODE_TOKENS
        + PLANNING_TOKENS
        + WHITESPACE_TOKENS
        + ASCII_CHARS
    )
    deduped: List[str] = []
    seen = set()
    for token in vocab:
        if token not in seen:
            deduped.append(token)
            seen.add(token)
    return deduped


def get_structured_special_tokens() -> List[str]:
    return list(dict.fromkeys(SPECIAL_TOKENS + STATE_MARKERS + LEGACY_ACTION_MARKERS + CANONICAL_ACTION_TOKENS + ACTION_TYPES))


WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NUMBER_RE = re.compile(r"-?\d+")


@dataclass
class StructuredTokenizer:
    vocab: List[str]

    def __post_init__(self) -> None:
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.vocab)}
        self.itos: Dict[int, str] = {i: tok for tok, i in self.stoi.items()}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.structured_tokens = sorted(
            [tok for tok in self.vocab if len(tok) > 1 and tok not in SPECIAL_TOKENS],
            key=len,
            reverse=True,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        i = 0
        while i < len(text):
            matched = False
            for token in self.structured_tokens:
                if text.startswith(token, i):
                    tokens.append(token)
                    i += len(token)
                    matched = True
                    break
            if matched:
                continue

            word_match = WORD_RE.match(text, i)
            if word_match:
                word = word_match.group(0)
                if word in self.stoi:
                    tokens.append(word)
                else:
                    tokens.extend(list(word))
                i = word_match.end()
                continue

            number_match = NUMBER_RE.match(text, i)
            if number_match:
                number = number_match.group(0)
                if number in self.stoi:
                    tokens.append(number)
                else:
                    tokens.extend(list(number))
                i = number_match.end()
                continue

            tokens.append(text[i])
            i += 1
        return tokens

    def token_to_id(self, token: str) -> int:
        return self.stoi.get(token, self.stoi.get("?", self.pad_id))

    def encode_unpadded(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for token in self.tokenize(text):
            ids.append(self.token_to_id(token))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode(self, text: str, seq_len: int) -> List[int]:
        ids = self.encode_unpadded(text, add_bos=True, add_eos=True)
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


CharTokenizer = StructuredTokenizer


def build_default_tokenizer() -> StructuredTokenizer:
    return StructuredTokenizer(vocab=_build_vocab())
