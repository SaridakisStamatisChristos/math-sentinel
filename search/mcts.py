
from __future__ import annotations

from typing import Any

from .beam import beam_search


def mcts_search(*args: Any, **kwargs: Any) -> Any:
    # Experimental placeholder: reuses beam search until a full MCTS policy/value split exists.
    return beam_search(*args, **kwargs)
