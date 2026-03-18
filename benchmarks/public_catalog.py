from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

from engine.task import ReasoningTask

from .base import BenchmarkSuite


ROOT = Path(__file__).resolve().parents[1]


def _annotate_case(case: ReasoningTask, suite_name: str, tier: str) -> ReasoningTask:
    meta = dict(case.meta)
    meta.setdefault("benchmark_suite", suite_name)
    meta.setdefault("benchmark_tier", tier)
    meta.setdefault("holdout_group", suite_name)
    meta.setdefault("source", "benchmark_claim_holdout")
    meta.setdefault("fixture_role", "benchmark")
    return ReasoningTask(
        task_id=case.task_id,
        domain=case.domain,
        prompt=case.prompt,
        answer=case.answer,
        goal=case.goal,
        meta=meta,
    )


def _swebench_fixture_case(name: str, prompt: str, patch: list[dict[str, str]], *, fixture_family: str) -> ReasoningTask:
    fixture_dir = ROOT / "benchmarks" / "fixtures" / fixture_family / name
    primary_file = patch[0]["path"] if patch else ""
    return ReasoningTask(
        task_id=f"swebench_{name}",
        domain="swebench_patch",
        prompt=prompt,
        answer="patched_and_verified",
        goal="Patch the repository so the tests pass",
        meta={
            "family": "swebench_patch",
            "fixture_dir": str(fixture_dir),
            "oracle_primary_file": primary_file,
            "oracle_patch": patch,
            "test_command": ["python", "-m", "unittest", "discover", "-s", "tests", "-q"],
        },
    )


def swebench_verified_smoke_suite() -> BenchmarkSuite:
    cases = [
        _swebench_fixture_case(
            "counter_bug",
            "Patch the repository so the failing tests pass. Fix the arithmetic bug in app.py and verify with the test suite.",
            [{"path": "app.py", "search": "    return a - b\n", "replace": "    return a + b\n"}],
            fixture_family="swebench_lite_smoke",
        ),
        _swebench_fixture_case(
            "slugify_bug",
            "Patch the repository so the failing tests pass. Fix slugify in text_utils.py and verify with the test suite.",
            [{"path": "text_utils.py", "search": '    return value.strip().lower().replace(" ", "_")\n', "replace": '    return "-".join(value.strip().lower().split())\n'}],
            fixture_family="swebench_lite_smoke",
        ),
    ]
    cases = [_annotate_case(case, "swebench_verified_smoke", "smoke") for case in cases]
    return BenchmarkSuite(
        name="swebench_verified_smoke",
        backend="swebench_ops",
        description="Local smoke suite shaped like SWE-bench Verified repository patch tasks.",
        tier="smoke",
        cases=cases,
    )


def swebench_verified_medium_suite() -> BenchmarkSuite:
    cases = [
        _swebench_fixture_case(
            "count_bug",
            "Patch the repository so the failing tests pass. The bug is only visible from the tests and the workspace has distractor files. Fix the counting behavior and verify with the test suite.",
            [{"path": "stats.py", "search": "    return len(items) - 1\n", "replace": "    return len(items)\n"}],
            fixture_family="swebench_medium_smoke",
        ),
        _swebench_fixture_case(
            "positive_filter_bug",
            "Patch the repository so the failing tests pass. Fix the positivity predicate so zero is excluded, and verify with the test suite.",
            [{"path": "filters.py", "search": ">= 0", "replace": "> 0"}],
            fixture_family="swebench_medium_smoke",
        ),
        _swebench_fixture_case(
            "slugify_bug_complex",
            "Patch the repository so the failing tests pass. Normalize slugs with hyphens instead of underscores and verify with the test suite.",
            [{"path": "text_utils.py", "search": '    return value.strip().lower().replace(" ", "_")\n', "replace": '    return "-".join(value.strip().lower().split())\n'}],
            fixture_family="swebench_medium_smoke",
        ),
        _swebench_fixture_case(
            "argument_order_bug",
            "Patch the repository so the failing tests pass. The helper is correct, but the public formatter passes arguments in the wrong order. Verify with the test suite.",
            [{"path": "formatters.py", "search": "    return build_name(last, first)\n", "replace": "    return build_name(first, last)\n"}],
            fixture_family="swebench_medium_smoke",
        ),
    ]
    cases = [_annotate_case(case, "swebench_verified_medium", "medium") for case in cases]
    return BenchmarkSuite(
        name="swebench_verified_medium",
        backend="swebench_ops",
        description="Harder local suite shaped like medium-difficulty SWE-bench Verified repository patch tasks.",
        tier="medium",
        cases=cases,
    )


def _gaia_fixture_case(
    case_id: str,
    domain: str,
    prompt: str,
    answer: str,
    recommended_tool: str,
    tool_input: str,
    fixture_family: str,
    fixture_subdir: str,
) -> ReasoningTask:
    fixture_dir = ROOT / "benchmarks" / "fixtures" / fixture_family / fixture_subdir
    evidence_file = tool_input.split("|", 1)[0] if "|" in tool_input else tool_input
    primary_evidence_file = evidence_file.split(",", 1)[0]
    return ReasoningTask(
        task_id=f"gaia_{case_id}",
        domain=domain,
        prompt=prompt,
        answer=answer,
        goal="Return the shortest correct final answer",
        meta={
            "family": domain,
            "fixture_dir": str(fixture_dir),
            "oracle_tool": recommended_tool,
            "oracle_input": tool_input,
            "oracle_evidence_file": primary_evidence_file,
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
            "gaia_smoke",
            "revenue_case",
        ),
        _gaia_fixture_case(
            "json_version",
            "gaia_json_reasoning",
            "Use the files in the workspace to answer this question: what is the latest release version recorded in report.json? Return only the version string.",
            "2.4.1",
            "json_path_lookup",
            "report.json|release.latest.version",
            "gaia_smoke",
            "report_case",
        ),
        _gaia_fixture_case(
            "meeting_slot",
            "gaia_schedule_reasoning",
            "Use the files in the workspace to answer this question: what is the earliest meeting slot where both Alice and Bob are available? Return only the slot label.",
            "10:30",
            "meeting_overlap",
            "schedule.json",
            "gaia_smoke",
            "meeting_case",
        ),
    ]
    cases = [_annotate_case(case, "gaia_smoke", "smoke") for case in cases]
    return BenchmarkSuite(
        name="gaia_smoke",
        backend="gaia_ops",
        description="Local smoke suite shaped like GAIA file-and-tool reasoning tasks.",
        tier="smoke",
        cases=cases,
    )


def gaia_medium_suite() -> BenchmarkSuite:
    cases = [
        _gaia_fixture_case(
            "cross_file_sales",
            "gaia_csv_reasoning",
            "Use the files in the workspace to answer this question: what is the total sales amount for the east region across q1_sales.csv and q2_sales.csv? Return only the number.",
            "41",
            "csv_region_total_multi",
            "q1_sales.csv,q2_sales.csv|region|east|amount",
            "gaia_medium_smoke",
            "cross_file_sales",
        ),
        _gaia_fixture_case(
            "city_leaderboard",
            "gaia_csv_reasoning",
            "Use the files in the workspace to answer this question: which city had the highest total amount in March 2025 in orders.csv? Return only the city name.",
            "Athens",
            "csv_groupby_max",
            "orders.csv|month=2025-03|group=city|amount|max",
            "gaia_medium_smoke",
            "city_leaderboard",
        ),
        _gaia_fixture_case(
            "roadmap_dates",
            "gaia_json_reasoning",
            "Use the files in the workspace to answer this question: in roadmap.json, which pending task owned by Mira has the earliest due date? Return only the task title.",
            "Finalize budget",
            "json_record_date_min",
            "roadmap.json|owner=Mira|status=pending|date=due|answer=title|min",
            "gaia_medium_smoke",
            "roadmap_dates",
        ),
    ]
    cases = [_annotate_case(case, "gaia_medium", "medium") for case in cases]
    return BenchmarkSuite(
        name="gaia_medium",
        backend="gaia_ops",
        description="Harder local suite shaped like medium-difficulty GAIA evidence-reasoning tasks.",
        tier="medium",
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
    cases = [_annotate_case(case, "math_public_smoke", "smoke") for case in cases]
    return BenchmarkSuite(
        name="math_public_smoke",
        backend="math",
        description="Local smoke suite for public-math-style evaluation hygiene.",
        tier="smoke",
        cases=cases,
    )


PUBLIC_SUITES: Dict[str, Callable[[], BenchmarkSuite]] = {
    "swebench_verified_smoke": swebench_verified_smoke_suite,
    "swebench_verified_medium": swebench_verified_medium_suite,
    "gaia_smoke": gaia_smoke_suite,
    "gaia_medium": gaia_medium_suite,
    "math_public_smoke": math_public_smoke_suite,
}


def available_public_suites(tier: str | None = None) -> list[str]:
    if tier is None:
        return list(PUBLIC_SUITES.keys())
    normalized = str(tier).strip().lower()
    return [name for name, builder in PUBLIC_SUITES.items() if builder().tier == normalized]


def available_public_suite_groups() -> Dict[str, list[str]]:
    return {
        "public_smoke": available_public_suites("smoke"),
        "public_medium": available_public_suites("medium"),
        "public_all": available_public_suites(),
    }


def load_public_suite(name: str) -> BenchmarkSuite:
    normalized = name.strip().lower()
    builder = PUBLIC_SUITES.get(normalized)
    if builder is None:
        raise ValueError(f"unknown public suite: {name}")
    return builder()
