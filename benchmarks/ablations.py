from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

from sentinel.config import deep_merge_dicts


DEFAULT_ABLATION_CATALOG = "config/benchmarks/ablation_matrix.yaml"


@dataclass
class BenchmarkAblation:
    name: str
    description: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_catalog(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    return payload


def load_ablation_catalog(path: str = DEFAULT_ABLATION_CATALOG) -> Dict[str, BenchmarkAblation]:
    catalog_path = Path(path)
    payload = _load_catalog(str(catalog_path))
    raw_ablations = payload.get("ablations", payload)
    ablations: Dict[str, BenchmarkAblation] = {}
    for name, item in raw_ablations.items():
        spec = dict(item or {})
        ablations[str(name)] = BenchmarkAblation(
            name=str(name),
            description=str(spec.get("description", name)),
            overrides=dict(spec.get("overrides", {})),
            tags=[str(entry) for entry in spec.get("tags", [])],
        )
    return ablations


def available_benchmark_ablations(path: str = DEFAULT_ABLATION_CATALOG) -> List[str]:
    return list(load_ablation_catalog(path).keys())


def resolve_benchmark_ablations(spec: str, path: str = DEFAULT_ABLATION_CATALOG) -> List[BenchmarkAblation]:
    catalog = load_ablation_catalog(path)
    normalized = (spec or "").strip().lower()
    if not normalized:
        return [catalog["baseline"]] if "baseline" in catalog else []
    if normalized == "all":
        return [catalog[name] for name in catalog]
    names = [name.strip() for name in spec.split(",") if name.strip()]
    resolved: List[BenchmarkAblation] = []
    for name in names:
        if name not in catalog:
            raise ValueError(f"unknown benchmark ablation: {name}")
        resolved.append(catalog[name])
    return resolved


def apply_benchmark_ablation(cfg: Dict[str, Any], ablation: BenchmarkAblation) -> Dict[str, Any]:
    if not ablation.overrides:
        return dict(cfg)
    return deep_merge_dicts(cfg, ablation.overrides)
