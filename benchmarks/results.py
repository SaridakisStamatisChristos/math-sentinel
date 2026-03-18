from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .base import BenchmarkSuiteResult


def suite_result_path(results_dir: str, suite_name: str) -> str:
    return str(Path(results_dir) / f"{suite_name}.json")


def save_suite_result(results_dir: str, result: BenchmarkSuiteResult) -> str:
    path = Path(suite_result_path(results_dir, result.suite))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    return str(path)


def load_suite_result(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
