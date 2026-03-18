from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def ensure_runtime_config_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    runtime_cfg = cfg.setdefault("runtime", {})
    runtime_cfg.setdefault("deterministic", False)
    runtime_cfg.setdefault("safe_mode", False)
    runtime_cfg.setdefault("structured_logs", True)
    runtime_cfg.setdefault("event_log_path", str(Path("logs") / "runtime_events.jsonl"))
    runtime_cfg.setdefault("fail_on_checkpoint_mismatch", False)
    return cfg


def ensure_benchmark_config_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    benchmark_cfg = cfg.setdefault("benchmark", {})
    benchmark_cfg.setdefault("assistance_mode", "unassisted")
    benchmark_cfg.setdefault("oracle_hints_enabled", False)
    return cfg


def apply_safe_runtime_profile(cfg: Dict[str, Any]) -> Dict[str, Any]:
    search_cfg = cfg.setdefault("search", {})
    search_cfg["decoder_mode"] = "strict"
    search_cfg["temperature"] = 0.0
    search_cfg["top_k"] = 1
    return cfg


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def device_from_cfg(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def configure_runtime(
    cfg: Dict[str, Any],
    *,
    deterministic_override: bool | None = None,
    safe_override: bool | None = None,
) -> str:
    ensure_runtime_config_defaults(cfg)
    if deterministic_override is not None:
        cfg["runtime"]["deterministic"] = bool(deterministic_override)
    if safe_override is not None:
        cfg["runtime"]["safe_mode"] = bool(safe_override)
    if bool(cfg["runtime"].get("safe_mode", False)):
        apply_safe_runtime_profile(cfg)
    set_seed(int(cfg.get("seed", 1337)), deterministic=bool(cfg["runtime"].get("deterministic", False)))
    return device_from_cfg(str(cfg.get("device", "auto")))
