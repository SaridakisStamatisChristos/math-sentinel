
from __future__ import annotations

from proof.state import ProofState


def clone_state(state: ProofState) -> ProofState:
    return state.clone()
