
from __future__ import annotations

from .generators import GeneratedTask


def build_gold_trace(task: GeneratedTask) -> str:
    p = task.prompt
    if task.domain == "arithmetic":
        op = task.meta["op"]
        tool = {"+" : "add", "-" : "subtract", "*" : "multiply"}[op]
        nums = f'{task.meta["a"]},{task.meta["b"]}'
        return (
            '<action type="THINK">Use exact integer arithmetic.</action>\n'
            f'<action type="APPLY" tool="{tool}">{nums}</action>\n'
            f'<action type="ANSWER">{task.answer}</action>\n'
            f'<answer>{task.answer}</answer>'
        )
    if task.domain == "fractions":
        frac = p.split(":", 1)[-1].strip()
        return (
            '<action type="THINK">Reduce numerator and denominator by their gcd.</action>\n'
            f'<action type="APPLY" tool="reduce_fraction">{frac}</action>\n'
            f'<action type="ANSWER">{task.answer}</action>\n'
            f'<answer>{task.answer}</answer>'
        )
    if task.domain == "divmod":
        nums = [x for x in p.split() if x.isdigit()]
        arg = ",".join(nums[:2]) if len(nums) >= 2 else p
        return (
            '<action type="THINK">Compute quotient and remainder.</action>\n'
            f'<action type="APPLY" tool="divmod">{arg}</action>\n'
            f'<action type="ANSWER">{task.answer}</action>\n'
            f'<answer>{task.answer}</answer>'
        )
    if task.domain == "gcd_lcm":
        nums = [x for x in p.split() if x.rstrip(",").isdigit()]
        arg = ",".join(n.strip(",") for n in nums[-2:])
        return (
            '<action type="THINK">Use gcd first, then derive lcm.</action>\n'
            f'<action type="APPLY" tool="gcd">{arg}</action>\n'
            f'<action type="APPLY" tool="lcm">{arg}</action>\n'
            f'<action type="ANSWER">{task.answer}</action>\n'
            f'<answer>{task.answer}</answer>'
        )
    if task.domain == "modular":
        nums = [x for x in p.replace("mod", "").split() if x.lstrip("-").isdigit()]
        arg = ",".join(nums[:2]) if len(nums) >= 2 else p
        return (
            '<action type="THINK">Reduce the integer modulo m.</action>\n'
            f'<action type="APPLY" tool="modular_reduce">{arg}</action>\n'
            f'<action type="ANSWER">{task.answer}</action>\n'
            f'<answer>{task.answer}</answer>'
        )
    if task.domain == "primality":
        n = "".join(ch for ch in p if ch.isdigit())
        return (
            '<action type="THINK">Check whether any divisor up to sqrt(n) exists.</action>\n'
            f'<action type="APPLY" tool="primality">{n}</action>\n'
            f'<action type="ANSWER">{task.answer}</action>\n'
            f'<answer>{task.answer}</answer>'
        )
    if task.domain == "factorization":
        n = "".join(ch for ch in p if ch.isdigit())
        return (
            '<action type="THINK">Extract prime divisors repeatedly.</action>\n'
            f'<action type="APPLY" tool="factorize">{n}</action>\n'
            f'<action type="ANSWER">{task.answer}</action>\n'
            f'<answer>{task.answer}</answer>'
        )
    if task.domain == "linear_equation":
        eq = p.split(":", 1)[-1].strip()
        return (
            '<action type="THINK">Isolate x using inverse operations.</action>\n'
            f'<action type="SUBGOAL">isolate x</action>\n'
            f'<action type="APPLY" tool="solve_linear_step">{eq}</action>\n'
            f'<action type="RESOLVE_SUBGOAL">isolate x</action>\n'
            f'<action type="ANSWER">{task.answer}</action>\n'
            f'<answer>{task.answer}</answer>'
        )
    if task.domain == "polynomial_simplify":
        expr = p.split(":", 1)[-1].strip()
        return (
            '<action type="THINK">Collect like terms.</action>\n'
            f'<action type="APPLY" tool="simplify_polynomial">{expr}</action>\n'
            f'<action type="ANSWER">{task.answer}</action>\n'
            f'<answer>{task.answer}</answer>'
        )
    if task.domain == "derivative":
        expr = p.split(":", 1)[-1].strip()
        return (
            '<action type="THINK">Differentiate term by term.</action>\n'
            f'<action type="APPLY" tool="derivative">{expr}</action>\n'
            f'<action type="ANSWER">{task.answer}</action>\n'
            f'<answer>{task.answer}</answer>'
        )
    if task.domain == "integral":
        expr = p.split(":", 1)[-1].strip()
        return (
            '<action type="THINK">Integrate term by term and add a constant.</action>\n'
            f'<action type="APPLY" tool="antiderivative">{expr}</action>\n'
            f'<action type="ANSWER">{task.answer}</action>\n'
            f'<answer>{task.answer}</answer>'
        )
    return (
        '<action type="THINK">Provide a short direct proof sketch.</action>\n'
        f'<action type="ANSWER">{task.answer}</action>\n'
        f'<answer>{task.answer}</answer>'
    )
