
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    prover: torch.nn.Module,
    verifier: torch.nn.Module,
    prover_optim: Optional[torch.optim.Optimizer],
    verifier_optim: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    step: int,
    config: Dict[str, Any],
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "step": step,
        "config": config,
        "prover": prover.state_dict(),
        "verifier": verifier.state_dict(),
        "prover_optim": prover_optim.state_dict() if prover_optim is not None else None,
        "verifier_optim": verifier_optim.state_dict() if verifier_optim is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extra_state": extra_state or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    prover: torch.nn.Module,
    verifier: torch.nn.Module,
    prover_optim: Optional[torch.optim.Optimizer] = None,
    verifier_optim: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    prover.load_state_dict(payload["prover"])
    verifier.load_state_dict(payload["verifier"])
    if prover_optim is not None and payload.get("prover_optim") is not None:
        prover_optim.load_state_dict(payload["prover_optim"])
    if verifier_optim is not None and payload.get("verifier_optim") is not None:
        verifier_optim.load_state_dict(payload["verifier_optim"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return payload
