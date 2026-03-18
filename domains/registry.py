from __future__ import annotations

from typing import Any

from domains.code_ops.backend import CodeOpsReasoningDomain
from domains.gaia_ops.backend import GaiaOpsReasoningDomain
from domains.math.backend import MathReasoningDomain
from domains.planning_ops.backend import PlanningOpsReasoningDomain
from domains.swebench_ops.backend import SwebenchOpsReasoningDomain
from domains.string_ops.backend import StringOpsReasoningDomain


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def available_backends() -> list[str]:
    return ["math", "string_ops", "code_ops", "planning_ops", "swebench_ops", "gaia_ops"]


def create_reasoning_domain(name: str, checker_plugin: str = "", runtime_config: dict[str, Any] | None = None) -> Any:
    normalized = _normalize_name(name or "math")
    if normalized == "math":
        return MathReasoningDomain(checker_plugin=checker_plugin)
    if normalized in {"string_ops", "strings", "text_ops"}:
        return StringOpsReasoningDomain()
    if normalized in {"code_ops", "code"}:
        return CodeOpsReasoningDomain()
    if normalized in {"planning_ops", "planning", "planner"}:
        return PlanningOpsReasoningDomain()
    if normalized in {"swebench_ops", "swebench", "swe_bench"}:
        return SwebenchOpsReasoningDomain(runtime_config=runtime_config)
    if normalized in {"gaia_ops", "gaia"}:
        return GaiaOpsReasoningDomain(runtime_config=runtime_config)
    raise ValueError(f"unknown backend: {name}")


def default_curriculum_config(name: str) -> str:
    normalized = _normalize_name(name or "math")
    if normalized == "math":
        return MathReasoningDomain.default_curriculum_config
    if normalized in {"string_ops", "strings", "text_ops"}:
        return StringOpsReasoningDomain.default_curriculum_config
    if normalized in {"code_ops", "code"}:
        return CodeOpsReasoningDomain.default_curriculum_config
    if normalized in {"planning_ops", "planning", "planner"}:
        return PlanningOpsReasoningDomain.default_curriculum_config
    if normalized in {"swebench_ops", "swebench", "swe_bench"}:
        return SwebenchOpsReasoningDomain.default_curriculum_config
    if normalized in {"gaia_ops", "gaia"}:
        return GaiaOpsReasoningDomain.default_curriculum_config
    raise ValueError(f"unknown backend: {name}")
