
from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Dict, List

try:
    import sympy as sp
except Exception:
    sp = None

from .rewrite import normalize_fraction_text, canonical_factorization, normalize_polynomial_text


def _eq_fraction(a: str, b: str) -> bool:
    try:
        fa = Fraction(a.strip())
        fb = Fraction(b.strip())
        return fa == fb
    except Exception:
        return normalize_fraction_text(a) == normalize_fraction_text(b)


def _eq_factorization(a: str, b: str) -> bool:
    return canonical_factorization(a) == canonical_factorization(b)


def _eq_pair_fields(a: str, b: str, names: List[str]) -> bool:
    def parse(s: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for name in names:
            m = re.search(rf"{name}\s*=\s*(-?\d+)", s)
            if m:
                out[name] = int(m.group(1))
        return out
    return parse(a) == parse(b)


def _eq_sympy_expr(a: str, b: str) -> bool:
    if sp is None:
        return normalize_polynomial_text(a) == normalize_polynomial_text(b)
    x = sp.Symbol("x")
    try:
        ea = sp.sympify(a, locals={"x": x})
        eb = sp.sympify(b, locals={"x": x})
        return sp.simplify(ea - eb) == 0
    except Exception:
        return normalize_polynomial_text(a) == normalize_polynomial_text(b)


def _eq_integral_up_to_constant(a: str, b: str) -> bool:
    if sp is None:
        return normalize_polynomial_text(a) == normalize_polynomial_text(b)
    x = sp.Symbol("x")
    try:
        ea = sp.sympify(a, locals={"x": x})
        eb = sp.sympify(b, locals={"x": x})
        diff = sp.expand(ea - eb)
        return diff.free_symbols <= set() or diff.diff(x) == 0
    except Exception:
        return False


def _eq_parity_proof(a: str, b: str) -> bool:
    a_low = a.lower()
    b_low = b.lower()
    required = ["even", "divisible by 2"]
    has_a = any(token in a_low for token in required) or "2*" in a_low or "2k" in a_low or "2*k" in a_low
    has_b = any(token in b_low for token in required) or "2*" in b_low or "2k" in b_low or "2*k" in b_low
    return has_a and has_b


def equivalent(domain: str, candidate: str, expected: str, meta: Dict[str, Any] | None = None) -> bool:
    meta = meta or {}
    family = meta.get("family", domain)
    if family in {"arithmetic", "modular", "primality", "linear_equation"}:
        return candidate.strip().replace(" ", "") == expected.strip().replace(" ", "")
    if family in {"fractions"}:
        return _eq_fraction(candidate, expected)
    if family in {"factorization"}:
        return _eq_factorization(candidate, expected)
    if family in {"divmod"}:
        return _eq_pair_fields(candidate, expected, ["q", "r"])
    if family in {"gcd_lcm"}:
        return _eq_pair_fields(candidate, expected, ["gcd", "lcm"])
    if family in {"derivative", "polynomial_simplify"}:
        return _eq_sympy_expr(candidate, expected)
    if family in {"integral"}:
        return _eq_integral_up_to_constant(candidate, expected)
    if family in {"parity_proof"}:
        return _eq_parity_proof(candidate, expected)
    return candidate.strip() == expected.strip()
