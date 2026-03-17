from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_runtime_config(config_path: str, search_config_path: str = "config/search.yaml") -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    search_path = Path(search_config_path)
    if not search_path.exists():
        return cfg

    raw_search_cfg = load_yaml(str(search_path))
    if not raw_search_cfg:
        return cfg

    search_cfg = raw_search_cfg.get("search", raw_search_cfg)
    merged_search = dict(cfg.get("search", {}))
    merged_search.update(search_cfg)
    cfg["search"] = merged_search
    return cfg
