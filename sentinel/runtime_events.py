from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .logging_utils import log_jsonl, now_ts


class RuntimeEventLogger:
    def __init__(self, path: str, *, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled and bool(path)

    def log(self, event: str, **payload: Any) -> None:
        if not self.enabled:
            return
        record: Dict[str, Any] = {"ts": now_ts(), "event": event}
        record.update(payload)
        log_jsonl(self.path, record)

    def __call__(self, event: str, **payload: Any) -> None:
        self.log(event, **payload)


def build_runtime_event_logger(cfg: Dict[str, Any]) -> RuntimeEventLogger:
    runtime_cfg = cfg.get("runtime", {})
    logs_dir = Path(cfg.get("paths", {}).get("logs_dir", "logs"))
    default_path = str(logs_dir / "runtime_events.jsonl")
    path = str(runtime_cfg.get("event_log_path", default_path))
    return RuntimeEventLogger(path, enabled=bool(runtime_cfg.get("structured_logs", True)))
