from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

from sentinel.config import deep_merge_dicts, load_runtime_config


DEFAULT_PROFILE_CATALOG = "config/benchmarks/profiles.yaml"


@dataclass
class BenchmarkProfile:
    name: str
    description: str
    config_path: str = ""
    overrides: Dict[str, Any] = field(default_factory=dict)
    suites: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_catalog(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    return payload


def load_profile_catalog(path: str = DEFAULT_PROFILE_CATALOG) -> Dict[str, BenchmarkProfile]:
    catalog_path = Path(path)
    payload = _load_catalog(str(catalog_path))
    raw_profiles = payload.get("profiles", payload)
    profiles: Dict[str, BenchmarkProfile] = {}
    for name, item in raw_profiles.items():
        spec = dict(item or {})
        config_ref = str(spec.get("config", "")).strip()
        config_path = ""
        if config_ref:
            config_path = str((catalog_path.parent / config_ref).resolve())
        profiles[str(name)] = BenchmarkProfile(
            name=str(name),
            description=str(spec.get("description", name)),
            config_path=config_path,
            overrides=dict(spec.get("overrides", {})),
            suites=[str(entry) for entry in spec.get("suites", [])],
            tags=[str(entry) for entry in spec.get("tags", [])],
            metadata=dict(spec.get("metadata", {})),
        )
    return profiles


def available_benchmark_profiles(path: str = DEFAULT_PROFILE_CATALOG) -> List[str]:
    return list(load_profile_catalog(path).keys())


def resolve_benchmark_profiles(spec: str, path: str = DEFAULT_PROFILE_CATALOG) -> List[BenchmarkProfile]:
    catalog = load_profile_catalog(path)
    normalized = (spec or "").strip().lower()
    if not normalized or normalized == "all":
        return [catalog[name] for name in catalog]
    names = [name.strip() for name in spec.split(",") if name.strip()]
    resolved: List[BenchmarkProfile] = []
    for name in names:
        if name not in catalog:
            raise ValueError(f"unknown benchmark profile: {name}")
        resolved.append(catalog[name])
    return resolved


def apply_benchmark_profile(base_cfg: Dict[str, Any], profile: BenchmarkProfile) -> Dict[str, Any]:
    cfg = deepcopy(base_cfg)
    if profile.config_path:
        cfg = load_runtime_config(profile.config_path, search_config_path="")
    if profile.overrides:
        cfg = deep_merge_dicts(cfg, profile.overrides)
    return cfg
