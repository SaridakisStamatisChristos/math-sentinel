
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_jsonl(path: str, record: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def compact_metrics(record: Dict[str, Any]) -> str:
    ordered = []
    for key in sorted(record.keys()):
        val = record[key]
        if isinstance(val, float):
            ordered.append(f"{key}={val:.4f}")
        else:
            ordered.append(f"{key}={val}")
    return " | ".join(ordered)
