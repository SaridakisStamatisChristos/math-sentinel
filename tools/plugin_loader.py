
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Optional


def load_plugin_module(path: str) -> Optional[ModuleType]:
    p = Path(path)
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location(p.stem, str(p))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
