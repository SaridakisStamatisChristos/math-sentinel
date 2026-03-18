from __future__ import annotations

from typing import Any, Dict, Iterable, List


ORACLE_PREFIX = "oracle_"
NON_ORACLE_RUNTIME_KEYS = {"oracle_hints_enabled"}


def is_runtime_oracle_field(key: str) -> bool:
    normalized = str(key).strip()
    return normalized.startswith(ORACLE_PREFIX) and normalized not in NON_ORACLE_RUNTIME_KEYS


def strip_oracle_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {str(key): value for key, value in metadata.items() if not is_runtime_oracle_field(str(key))}


def ensure_benchmark_audit(metadata: Dict[str, Any], *, assistance_mode: str = "unassisted") -> Dict[str, Any]:
    audit = metadata.get("benchmark_audit", {})
    if not isinstance(audit, dict):
        audit = {}
    audit.setdefault("assistance_mode", str(assistance_mode))
    audit.setdefault("oracle_fields_touched", [])
    audit.setdefault("integrity_events", [])
    audit.setdefault("guided_rollout_used", False)
    audit.setdefault("guided_rollout_steps", 0)
    audit.setdefault("fallback_repair_used", False)
    audit.setdefault("fallback_repair_attempts", 0)
    audit.setdefault("oracle_fields_present_in_runtime", False)
    audit.setdefault("benchmark_integrity_passed", True)
    metadata["benchmark_audit"] = audit
    return audit


def _append_unique(values: List[str], item: str) -> None:
    text = str(item).strip()
    if text and text not in values:
        values.append(text)


def record_integrity_event(metadata: Dict[str, Any], event: str, **payload: Any) -> Dict[str, Any]:
    assistance_mode = str(metadata.get("benchmark_assistance_mode", metadata.get("assistance_mode", "unassisted")))
    audit = ensure_benchmark_audit(metadata, assistance_mode=assistance_mode)
    rendered = str(event).strip()
    details = ", ".join(f"{key}={value}" for key, value in sorted(payload.items()) if str(value).strip())
    if details:
        rendered = f"{rendered}: {details}"
    _append_unique(audit["integrity_events"], rendered)
    if str(event).strip().startswith("oracle_"):
        audit["benchmark_integrity_passed"] = False
    return audit


def mark_oracle_field_touched(metadata: Dict[str, Any], field: str, *, reason: str) -> Dict[str, Any]:
    assistance_mode = str(metadata.get("benchmark_assistance_mode", metadata.get("assistance_mode", "unassisted")))
    audit = ensure_benchmark_audit(metadata, assistance_mode=assistance_mode)
    _append_unique(audit["oracle_fields_touched"], field)
    audit["benchmark_integrity_passed"] = False
    record_integrity_event(metadata, "oracle_field_touched", field=field, reason=reason)
    return audit


def mark_runtime_oracle_fields_present(metadata: Dict[str, Any], fields: Iterable[str]) -> Dict[str, Any]:
    assistance_mode = str(metadata.get("benchmark_assistance_mode", metadata.get("assistance_mode", "unassisted")))
    audit = ensure_benchmark_audit(metadata, assistance_mode=assistance_mode)
    field_list = [str(field).strip() for field in fields if str(field).strip()]
    if not field_list:
        return audit
    audit["oracle_fields_present_in_runtime"] = True
    audit["benchmark_integrity_passed"] = False
    record_integrity_event(metadata, "oracle_fields_present_in_runtime", fields="|".join(field_list))
    return audit


def collect_state_audit(metadata: Dict[str, Any]) -> Dict[str, Any]:
    assistance_mode = str(metadata.get("benchmark_assistance_mode", metadata.get("assistance_mode", "unassisted")))
    audit = ensure_benchmark_audit(metadata, assistance_mode=assistance_mode)
    runtime_oracle_fields = [str(key) for key in metadata.keys() if is_runtime_oracle_field(str(key))]
    if assistance_mode == "unassisted" and runtime_oracle_fields:
        mark_runtime_oracle_fields_present(metadata, runtime_oracle_fields)
    return audit
