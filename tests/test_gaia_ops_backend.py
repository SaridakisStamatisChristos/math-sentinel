from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from domains.gaia_ops.backend import (
    GaiaOpsReasoningDomain,
    _extract_usgs_collection_locations,
    _infer_xlsx_answer,
    _nature_article_type_counts,
    _solve_arxiv_overlap,
    _solve_ping_pong_choice,
    _solve_unlambda_missing_token,
)
from engine.task import ReasoningTask


class GaiaOpsBackendTests(unittest.TestCase):
    def test_evidence_driven_fallback_loop_solves_csv_case_without_oracle_tool_hints(self) -> None:
        backend = GaiaOpsReasoningDomain()
        task = backend.benchmark_tasks()[0]
        task.meta.pop("oracle_evidence_file", None)
        task.meta.pop("oracle_tool", None)
        task.meta.pop("oracle_input", None)
        state = backend.make_state(task)
        executor = backend.create_executor()

        info = {}
        for _ in range(6):
            repair = backend.fallback_repairs(state)[0]
            state, info = executor.apply(state, repair)
            if state.metadata.get("candidate_answer"):
                answer_action = backend.fallback_repairs(state)[0]
                state, info = executor.apply(state, answer_action)
            if state.status == "solved":
                break

        self.assertEqual(state.status, "solved")
        self.assertEqual(state.final_answer, "22")
        self.assertIn("22", " ".join(state.evidence_refs + state.derived_facts))
        self.assertGreaterEqual(float(info["goal_progress"]), 0.8)
        self.assertEqual(state.metadata.get("target_file"), "sales.csv")

    def test_benchmark_tasks_cover_multiple_reasoning_families(self) -> None:
        backend = GaiaOpsReasoningDomain()
        families = {task.domain for task in backend.benchmark_tasks()}

        self.assertEqual(families, {"gaia_csv_reasoning", "gaia_json_reasoning", "gaia_schedule_reasoning"})

    def test_manual_task_matches_fixture_case_for_sample_flow(self) -> None:
        backend = GaiaOpsReasoningDomain()

        task = backend.manual_task(
            "gaia_csv_reasoning",
            "Use the files in the workspace to answer this question: what is the total sales amount for the east region in sales.csv? Return only the number.",
        )

        self.assertIn("fixture_dir", task.meta)
        state = backend.make_state(task)
        self.assertIn("sales.csv", state.metadata["workspace_files"])
        self.assertEqual(state.metadata.get("target_file"), "sales.csv")

    def test_unassisted_runtime_state_strips_oracle_metadata(self) -> None:
        backend = GaiaOpsReasoningDomain()
        task = backend.benchmark_tasks()[0]

        state = backend.make_state(task)

        self.assertNotIn("oracle_evidence_file", state.metadata)
        self.assertNotIn("oracle_tool", state.metadata)
        self.assertNotIn("oracle_input", state.metadata)
        self.assertEqual(state.metadata.get("benchmark_audit", {}).get("assistance_mode"), "unassisted")

    def test_medium_fixture_fallback_loop_solves_cross_file_sales_case(self) -> None:
        backend = GaiaOpsReasoningDomain()
        task = next(task for task in backend.benchmark_tasks() if task.task_id == "gaia_cross_file_sales")
        state = backend.make_state(task)
        executor = backend.create_executor()

        for _ in range(6):
            repair = backend.fallback_repairs(state)[0]
            state, _ = executor.apply(state, repair)
            if state.metadata.get("candidate_answer"):
                answer_action = backend.fallback_repairs(state)[0]
                state, _ = executor.apply(state, answer_action)
            if state.status == "solved":
                break

        self.assertEqual(state.status, "solved")
        self.assertEqual(state.final_answer, "41")

    def test_inspect_and_solve_populate_evidence_graph_and_confidence(self) -> None:
        backend = GaiaOpsReasoningDomain()
        task = backend.benchmark_tasks()[0]
        state = backend.make_state(task)
        executor = backend.create_executor()

        for _ in range(4):
            repair = backend.fallback_repairs(state)[0]
            state, _ = executor.apply(state, repair)
            if state.metadata.get("candidate_answer"):
                break

        graph = state.metadata.get("evidence_graph", {})

        self.assertTrue(graph.get("files"))
        self.assertGreaterEqual(float(state.metadata.get("answer_confidence", 0.0)), 0.45)
        self.assertTrue(state.metadata.get("answer_provenance"))

    @patch("domains.gaia_ops.backend._arxiv_search")
    def test_external_arxiv_research_flow_solves_overlap_question(self, mock_search: object) -> None:
        mock_search.side_effect = [
            [
                {
                    "title": "Fairness in Agreement With European Values: An Interdisciplinary Perspective on AI Regulation",
                    "summary": "We discuss standardized, localized, utilitarian, egalitarian, consequential, and deontological conceptions of fairness.",
                    "published": "2022-06-15T00:00:00Z",
                    "id": "a",
                    "categories": ["cs.AI"],
                }
            ],
            [
                {
                    "title": "Example Physics and Society article",
                    "summary": "The paper describes an egalitarian society under resource constraints.",
                    "published": "2016-08-11T00:00:00Z",
                    "id": "b",
                    "categories": ["physics.soc-ph"],
                }
            ],
        ]
        backend = GaiaOpsReasoningDomain()
        task = ReasoningTask(
            task_id="gaia_external_arxiv_case",
            domain="gaia_json_reasoning",
            prompt="A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes, where each axis has a label word at both ends. Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?",
            answer="egalitarian",
            goal="Return the shortest correct final answer",
            meta={"family": "gaia_json_reasoning"},
        )
        state = backend.make_state(task)
        executor = backend.create_executor()

        for _ in range(5):
            repair = backend.fallback_repairs(state)[0]
            state, _ = executor.apply(state, repair)
            if state.metadata.get("candidate_answer"):
                answer_action = backend.fallback_repairs(state)[0]
                state, _ = executor.apply(state, answer_action)
                break

        self.assertEqual(state.status, "solved")
        self.assertEqual(state.final_answer, "egalitarian")

    def test_solve_arxiv_overlap_returns_matching_term(self) -> None:
        primary = [{"title": "AI regulation", "summary": "standardized, localized, utilitarian, egalitarian, consequential and deontological", "published": "", "id": "", "categories": []}]
        secondary = [{"title": "Physics and Society", "summary": "An egalitarian society is discussed here", "published": "", "id": "", "categories": []}]

        answer, evidence = _solve_arxiv_overlap(primary, secondary)

        self.assertEqual(answer, "egalitarian")
        self.assertTrue(evidence)

    def test_unlambda_solver_identifies_missing_backtick(self) -> None:
        answer, evidence = _solve_unlambda_missing_token(
            'In Unlambda, what exact charcter or text needs to be added to correct the following code to output "For penguins"? '
            'Code:\n\n`r```````````.F.o.r. .p.e.n.g.u.i.n.si'
        )

        self.assertEqual(answer, "backtick")
        self.assertTrue(evidence)

    def test_ping_pong_solver_prefers_ball_three(self) -> None:
        answer, evidence = _solve_ping_pong_choice()

        self.assertEqual(answer, "3")
        self.assertTrue(any("best ball 3" in item for item in evidence))

    def test_infer_xlsx_answer_returns_oldest_bluray_title(self) -> None:
        path = next(
            Path("data/official_corpus/gaia/attachments/32102e3e-d12a-4209-9163-7b3a104efe5d").glob("*.xlsx")
        )
        prompt = (
            "The attached spreadsheet shows the inventory for a movie and video game rental store in Seattle, Washington. "
            "What is the title of the oldest Blu-Ray recorded in this spreadsheet? Return it as appearing in the spreadsheet."
        )

        answer, evidence = _infer_xlsx_answer(prompt, path)

        self.assertEqual(answer, "Time-Parking 2: Parallel Universe")
        self.assertTrue(evidence)

    def test_nature_article_type_counts_parse_html(self) -> None:
        html = """
        <span class="c-meta__type">Article</span>
        <span class="c-meta__type">Article</span>
        <span class="c-meta__type">Matters Arising</span>
        """

        counts = _nature_article_type_counts(html)

        self.assertEqual(counts["Article"], 2)
        self.assertEqual(counts["Matters Arising"], 1)

    def test_extract_usgs_collection_locations_parses_county_locality_and_year(self) -> None:
        text = "Pinellas Gulf of America, Florida, Fred Howard Park 2018 03100207 Crystal-Pithlachascotee eradicated"

        records = _extract_usgs_collection_locations(text)

        self.assertEqual(records, [{"county": "Pinellas", "locality": "Fred Howard Park", "year": "2018"}])
