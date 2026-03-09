
from __future__ import annotations

import re
from fractions import Fraction
from typing import List


def normalize_fraction_text(text: str) -> str:
    text = text.strip().replace(" ", "")
    if "/" in text:
        a, b = text.split("/", 1)
        try:
            frac = Fraction(int(a), int(b))
            return f"{frac.numerator}/{frac.denominator}"
        except Exception:
            return text
    return text


def canonical_factorization(text: str) -> str:
    nums: List[int] = []
    for token in re.split(r"[^0-9]+", text):
        if token:
            nums.append(int(token))
    nums.sort()
    return "*".join(str(x) for x in nums) if nums else text.strip()


def normalize_polynomial_text(text: str) -> str:
    text = text.replace(" ", "").replace("+-", "-")
    text = re.sub(r"\b1x", "x", text)
    text = re.sub(r"\+\-", "-", text)
    return text
