from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

from .model_backends import ensure_model_config_defaults
from .runtime import apply_safe_runtime_profile, ensure_runtime_config_defaults


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return payload or {}


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml_resolved(path: str, _seen: set[str] | None = None) -> Dict[str, Any]:
    path_obj = Path(path).resolve()
    seen = set(_seen or set())
    if str(path_obj) in seen:
        raise ValueError(f"cyclic config extends detected for {path_obj}")
    seen.add(str(path_obj))

    payload = dict(load_yaml(str(path_obj)))
    parent_ref = payload.pop("extends", None)
    if not parent_ref:
        return payload

    parent_path = (path_obj.parent / str(parent_ref)).resolve()
    parent_payload = load_yaml_resolved(str(parent_path), seen)
    return deep_merge_dicts(parent_payload, payload)


def load_runtime_config(config_path: str, search_config_path: str = "config/search.yaml") -> Dict[str, Any]:
    cfg = load_yaml_resolved(config_path)
    search_path = Path(search_config_path) if search_config_path else None
    if search_path is None or not search_path.exists():
        cfg = ensure_model_config_defaults(cfg)
        cfg = ensure_runtime_config_defaults(cfg)
        if bool(cfg.get("runtime", {}).get("safe_mode", False)):
            apply_safe_runtime_profile(cfg)
        return cfg

    raw_search_cfg = load_yaml_resolved(str(search_path))
    if not raw_search_cfg:
        cfg = ensure_model_config_defaults(cfg)
        cfg = ensure_runtime_config_defaults(cfg)
        if bool(cfg.get("runtime", {}).get("safe_mode", False)):
            apply_safe_runtime_profile(cfg)
        return cfg

    search_cfg = raw_search_cfg.get("search", raw_search_cfg)
    cfg["search"] = deep_merge_dicts(dict(cfg.get("search", {})), search_cfg)
    cfg = ensure_model_config_defaults(cfg)
    cfg = ensure_runtime_config_defaults(cfg)
    if bool(cfg.get("runtime", {}).get("safe_mode", False)):
        apply_safe_runtime_profile(cfg)
    return cfg
