
from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

try:
    import sympy as sp
except Exception:
    sp = None
from engine.task import ReasoningTask


GeneratedTask = ReasoningTask


def _rid(prefix: str) -> str:
    return f"{prefix}_{random.randint(10**7, 10**8-1)}"


def gen_arithmetic() -> GeneratedTask:
    a, b = random.randint(-99, 99), random.randint(-99, 99)
    op = random.choice(["+", "-", "*"])
    prompt = f"Compute: {a} {op} {b}"
    ans = str(eval(f"{a}{op}{b}"))
    return GeneratedTask(_rid("arith"), "arithmetic", prompt, ans, "Compute the integer result", {"family": "arithmetic", "a": a, "b": b, "op": op})


def gen_fractions() -> GeneratedTask:
    a = random.randint(1, 25)
    b = random.randint(1, 25)
    num = a * random.randint(1, 9)
    den = b * random.randint(1, 9)
    from fractions import Fraction
    f = Fraction(num, den)
    prompt = f"Reduce the fraction: {num}/{den}"
    return GeneratedTask(_rid("frac"), "fractions", prompt, f"{f.numerator}/{f.denominator}", "Reduce to lowest terms", {"family": "fractions"})


def gen_divmod() -> GeneratedTask:
    a = random.randint(1, 200)
    b = random.randint(1, 20)
    q, r = divmod(a, b)
    prompt = f"Compute quotient and remainder for {a} divided by {b}"
    return GeneratedTask(_rid("divmod"), "divmod", prompt, f"q={q}, r={r}", "Return q and r", {"family": "divmod"})


def gen_gcd_lcm() -> GeneratedTask:
    a = random.randint(2, 120)
    b = random.randint(2, 120)
    import math
    g = math.gcd(a, b)
    l = abs(a * b) // g
    prompt = f"Compute gcd and lcm of {a} and {b}"
    return GeneratedTask(_rid("gcd"), "gcd_lcm", prompt, f"gcd={g}, lcm={l}", "Return gcd and lcm", {"family": "gcd_lcm"})


def gen_modular() -> GeneratedTask:
    a = random.randint(-200, 200)
    m = random.randint(2, 19)
    prompt = f"Compute {a} mod {m}"
    return GeneratedTask(_rid("mod"), "modular", prompt, str(a % m), "Return the least nonnegative residue", {"family": "modular"})


def gen_primality() -> GeneratedTask:
    n = random.randint(2, 200)
    d = 2
    is_prime = True
    while d * d <= n:
        if n % d == 0:
            is_prime = False
            break
        d += 1
    prompt = f"Classify {n} as prime or composite"
    return GeneratedTask(_rid("prime"), "primality", prompt, "prime" if is_prime else "composite", "Return prime or composite", {"family": "primality"})


def gen_factorization() -> GeneratedTask:
    n = random.randint(2, 200)
    x = n
    factors: List[int] = []
    d = 2
    while d * d <= x:
        while x % d == 0:
            factors.append(d)
            x //= d
        d += 1
    if x > 1:
        factors.append(x)
    prompt = f"Prime-factorize {n}"
    return GeneratedTask(_rid("fact"), "factorization", prompt, "*".join(map(str, factors)), "Return the prime factorization", {"family": "factorization"})


def gen_linear_equation() -> GeneratedTask:
    a = random.choice([i for i in range(-8, 9) if i not in {0}])
    x = random.randint(-9, 9)
    b = random.randint(-20, 20)
    c = a * x + b
    prompt = f"Solve: {a}x + {b} = {c}"
    return GeneratedTask(_rid("lin"), "linear_equation", prompt, f"x={x}", "Solve for x", {"family": "linear_equation", "a": a, "b": b, "c": c})


def gen_polynomial_simplify() -> GeneratedTask:
    a, b, c = random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)
    expr = f"({a}*x + {b}) + ({c}*x)"
    if sp is not None:
        x = sp.Symbol("x")
        ans = str(sp.expand(sp.sympify(expr, locals={'x': x})))
    else:
        ans = f"{a+c}*x+{b}"
    prompt = f"Simplify polynomial: {expr}"
    return GeneratedTask(_rid("poly"), "polynomial_simplify", prompt, ans, "Simplify the expression", {"family": "polynomial_simplify"})


def _random_poly() -> Tuple[str, str, str]:
    coeffs = [random.randint(-4, 4) for _ in range(4)]
    coeffs[-1] = random.choice([1, 2, 3, 4])
    if sp is not None:
        x = sp.Symbol("x")
        expr = sum(sp.Integer(coeffs[i]) * x**i for i in range(len(coeffs)))
        derivative = str(sp.expand(sp.diff(expr, x)))
        integral = str(sp.expand(sp.integrate(expr, x))) + " + C"
        return str(sp.expand(expr)), derivative, integral
    expr = " + ".join(f"{c}*x^{i}" if i > 1 else (f"{c}*x" if i == 1 else str(c)) for i, c in enumerate(coeffs))
    return expr, expr, expr + " + C"


def gen_derivative() -> GeneratedTask:
    expr, derivative, _ = _random_poly()
    prompt = f"Differentiate with respect to x: {expr}"
    return GeneratedTask(_rid("der"), "derivative", prompt, derivative, "Return d/dx", {"family": "derivative"})


def gen_integral() -> GeneratedTask:
    expr, _, integral = _random_poly()
    prompt = f"Find an antiderivative with respect to x: {expr}"
    return GeneratedTask(_rid("int"), "integral", prompt, integral, "Return any antiderivative", {"family": "integral"})


def gen_parity_proof() -> GeneratedTask:
    n = random.randint(1, 20)
    prompt = f"Give a short proof sketch that {2*n} is even."
    answer = "Because 2n = 2*k for k=n, it is divisible by 2 and therefore even."
    return GeneratedTask(_rid("parity"), "parity_proof", prompt, answer, "Provide a short proof sketch", {"family": "parity_proof"})


GENERATORS = {
    "arithmetic": gen_arithmetic,
    "fractions": gen_fractions,
    "divmod": gen_divmod,
    "gcd_lcm": gen_gcd_lcm,
    "modular": gen_modular,
    "primality": gen_primality,
    "factorization": gen_factorization,
    "linear_equation": gen_linear_equation,
    "polynomial_simplify": gen_polynomial_simplify,
    "derivative": gen_derivative,
    "integral": gen_integral,
    "parity_proof": gen_parity_proof,
}


def sample_task(domains: List[str]) -> GeneratedTask:
    domain = random.choice(domains)
    return GENERATORS[domain]()
