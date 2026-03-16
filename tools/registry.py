
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from . import arithmetic, algebra, calculus, number_theory, fractions, logic, sympy_bridge
from .plugin_loader import load_plugin_module


ToolFn = Callable[[str, Any], Dict[str, Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: Dict[str, ToolFn] = {}
        self.register_builtins()

    def register(self, name: str, fn: ToolFn) -> None:
        self.tools[name] = fn

    def register_builtins(self) -> None:
        self.register("add", arithmetic.add)
        self.register("subtract", arithmetic.subtract)
        self.register("multiply", arithmetic.multiply)
        self.register("divmod", arithmetic.divmod_tool)
        self.register("reduce_fraction", fractions.reduce_fraction)
        self.register("compare_fractions", fractions.compare_fractions)
        self.register("common_denominator", fractions.common_denominator)
        self.register("gcd", number_theory.gcd_tool)
        self.register("lcm", number_theory.lcm_tool)
        self.register("gcd_lcm", number_theory.gcd_lcm)
        self.register("primality", number_theory.primality)
        self.register("factorize", number_theory.factorize)
        self.register("modular_reduce", number_theory.modular_reduce)
        self.register("solve_linear_step", algebra.solve_linear_step)
        self.register("simplify_polynomial", algebra.simplify_polynomial)
        self.register("expand_or_factor", algebra.expand_or_factor)
        self.register("normalize_expression", algebra.normalize_expression)
        self.register("derivative", calculus.derivative)
        self.register("antiderivative", calculus.antiderivative)
        self.register("simplify_calculus_form", calculus.simplify_calculus_form)
        self.register("equality_transitivity", logic.equality_transitivity)
        self.register("contradiction_marker", logic.contradiction_marker)
        self.register("prove_even", logic.prove_even)
        if sympy_bridge.available():
            self.register("sympy_simplify", sympy_bridge.simplify_expr)
            self.register("sympy_equivalent", sympy_bridge.equivalent_expr)

    def call(self, name: str, arg: str, state: Any = None) -> Dict[str, Any]:
        fn = self.tools.get(name)
        if fn is None:
            return {"ok": False, "result": f"unknown tool: {name}"}
        return fn(arg, state)

    def load_plugin(self, path: str) -> bool:
        module = load_plugin_module(path)
        if module is None:
            return False
        if hasattr(module, "register"):
            module.register(self)
            return True
        if hasattr(module, "TOOL_FUNCS"):
            for name, fn in getattr(module, "TOOL_FUNCS").items():
                self.register(name, fn)
            return True
        return False
