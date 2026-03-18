from __future__ import annotations

from pathlib import Path
from typing import Any

from engine.task import ReasoningTask

from .base import BenchmarkSuite


ROOT = Path(__file__).resolve().parents[1]


def _swebench_fixture_case(name: str, prompt: str, patch: list[dict[str, str]]) -> ReasoningTask:
    fixture_dir = ROOT / "benchmarks" / "fixtures" / "swebench_lite_smoke" / name
    return ReasoningTask(
        task_id=f"swebench_{name}",
        domain="swebench_patch",
        prompt=prompt,
        answer="patched_and_verified",
        goal="Patch the repository so the tests pass",
        meta={
            "family": "swebench_patch",
            "fixture_dir": str(fixture_dir),
            "gold_patch": patch,
            "test_command": ["python", "-m", "unittest", "discover", "-s", "tests", "-q"],
        },
    )


def swebench_verified_smoke_suite() -> BenchmarkSuite:
    cases = [
        _swebench_fixture_case(
            "counter_bug",
            "Patch the repository so the failing tests pass. Fix the arithmetic bug in app.py and verify with the test suite.",
            [{"path": "app.py", "search": "    return a - b\n", "replace": "    return a + b\n"}],
        ),
        _swebench_fixture_case(
            "slugify_bug",
            "Patch the repository so the failing tests pass. Fix slugify in text_utils.py and verify with the test suite.",
            [{"path": "text_utils.py", "search": '    return value.strip().lower().replace(" ", "_")\n', "replace": '    return "-".join(value.strip().lower().split())\n'}],
        ),
    ]
    return BenchmarkSuite(
        name="swebench_verified_smoke",
        backend="swebench_ops",
        description="Local smoke suite shaped like SWE-bench Verified repository patch tasks.",
        tier="smoke",
        cases=cases,
    )


def _gaia_fixture_case(case_id: str, domain: str, prompt: str, answer: str, recommended_tool: str, tool_input: str, fixture_subdir: str) -> ReasoningTask:
    fixture_dir = ROOT / "benchmarks" / "fixtures" / "gaia_smoke" / fixture_subdir
    return ReasoningTask(
        task_id=f"gaia_{case_id}",
        domain=domain,
        prompt=prompt,
        answer=answer,
        goal="Return the shortest correct final answer",
        meta={
            "family": domain,
            "fixture_dir": str(fixture_dir),
            "recommended_tool": recommended_tool,
            "tool_input": tool_input,
        },
    )


def gaia_smoke_suite() -> BenchmarkSuite:
    cases = [
        _gaia_fixture_case(
            "csv_total",
            "gaia_csv_reasoning",
            "Use the files in the workspace to answer this question: what is the total sales amount for the east region in sales.csv? Return only the number.",
            "22",
            "csv_region_total",
            "sales.csv|region|east|amount",
            "revenue_case",
        ),
        _gaia_fixture_case(
            "json_version",
            "gaia_json_reasoning",
            "Use the files in the workspace to answer this question: what is the latest release version recorded in report.json? Return only the version string.",
            "2.4.1",
            "json_path_lookup",
            "report.json|release.latest.version",
            "report_case",
        ),
        _gaia_fixture_case(
            "meeting_slot",
            "gaia_schedule_reasoning",
            "Use the files in the workspace to answer this question: what is the earliest meeting slot where both Alice and Bob are available? Return only the slot label.",
            "10:30",
            "meeting_overlap",
            "schedule.json",
            "meeting_case",
        ),
    ]
    return BenchmarkSuite(
        name="gaia_smoke",
        backend="gaia_ops",
        description="Local smoke suite shaped like GAIA file-and-tool reasoning tasks.",
        tier="smoke",
        cases=cases,
    )


def math_public_smoke_suite() -> BenchmarkSuite:
    cases = [
        ReasoningTask(
            task_id="math_public_arith",
            domain="arithmetic",
            prompt="Compute: 13 + 29",
            answer="42",
            goal="Compute the result",
            meta={"family": "arithmetic"},
        ),
        ReasoningTask(
            task_id="math_public_linear",
            domain="linear_equation",
            prompt="Solve: 5x - 10 = 0",
            answer="x=2",
            goal="Solve the equation",
            meta={"family": "linear_equation"},
        ),
    ]
    return BenchmarkSuite(
        name="math_public_smoke",
        backend="math",
        description="Local smoke suite for public-math-style evaluation hygiene.",
        tier="smoke",
        cases=cases,
    )


def available_public_suites() -> list[str]:
    return ["swebench_verified_smoke", "gaia_smoke", "math_public_smoke"]


def load_public_suite(name: str) -> BenchmarkSuite:
    normalized = name.strip().lower()
    if normalized == "swebench_verified_smoke":
        return swebench_verified_smoke_suite()
    if normalized == "gaia_smoke":
        return gaia_smoke_suite()
    if normalized == "math_public_smoke":
        return math_public_smoke_suite()
    raise ValueError(f"unknown public suite: {name}")
