from __future__ import annotations

import json
import shutil
import statistics
import unittest
import urllib.error
from pathlib import Path
from typing import Any, cast
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont

from domains.gaia_ops.backend import (
    GaiaOpsReasoningDomain,
    _best_person_name_from_documents,
    _build_plan_metadata,
    _classify_no_file_source_families,
    _extract_special_research_plan,
    _extract_usgs_collection_locations,
    _generalized_browse_candidate_bundles,
    _infer_csv_answer,
    _infer_text_answer,
    _infer_xlsx_answer,
    _nature_article_type_counts,
    _pdf_text_from_url,
    _solve_advanced_spreadsheet_ops,
    _solve_arxiv_overlap,
    _solve_author_prior_publication,
    _solve_benjerry_background_rhyme,
    _solve_literal_word_instruction,
    _solve_colored_number_statistics_image,
    _solve_elisa_ec_numbers,
    _solve_github_public_artifact_ops,
    _solve_image_vision_ops,
    _solve_audio_transcription_ops,
    _solve_cross_source_entity_ops,
    _solve_office_document_ops,
    _solve_paper_numeric_lookup,
    _solve_paper_compare_ops,
    _solve_scholarly_reference_ops,
    _solve_pubchem_food_additive_transformations,
    _solve_historical_reference_navigation_ops,
    _solve_public_record_ops,
    _public_record_search_documents,
    _public_record_search_queries,
    _parse_service_daily_metric_line,
    _solve_public_record_schedule_arrival_time,
    _solve_public_reference_history_ops,
    _solve_broad_symbolic_ops,
    _solve_replit_vscode_command,
    _solve_public_scalar_transform_ops,
    _solve_reversed_instruction,
    _easyocr_text_lines_with_variants,
    _select_best_solver_candidate,
    _solve_thinking_machine_prediction,
    _solve_unlambda_missing_token,
    _solve_generic_public_reference,
    _fetch_benjerry_graveyard_entries,
    _fetch_search_documents,
    _first_citation_reference_url,
    _best_quiz_arithmetic_match,
    _best_quiz_improper_to_mixed_match,
    _best_quiz_mixed_to_improper_match,
    _browser_fetch_dom,
    _http_get_text,
    _http_get_text_cached,
    _solve_orcid_average_from_jsonld,
    _solve_usda_standards_supersession,
    _solve_video_transcript_ops,
    _solve_butterfat_nutrition_image,
    _solve_embedded_sorting_code_image,
    _solver_candidate_bundle,
    _validate_candidate_answer,
    _infer_fraction_pair_from_ocr_sequences,
    _green_step_area,
    _round_segments_to_total,
    _solve_storage_plan_upgrade_math,
    _solve_web_archive_ops,
    _wikipedia_bidirectional_link_distance,
    _wikipedia_page_links_as_of,
    _solve_wikipedia_link_distance,
    _solve_wikipedia_revision_count,
    _solve_youtube_bird_species_count,
    _page_image_urls,
    _search_documents_for_title,
    _extract_historical_navigation_title,
    _search_documents_from_prompt,
    _temporal_anchor,
    _temporal_query_variants,
    plan_question,
    solve_question,
)
from engine.actions import Action, ActionType
from engine.state import ReasoningState
from engine.task import ReasoningTask


class GaiaOpsBackendTests(unittest.TestCase):
    @patch("domains.gaia_ops.backend.urllib.request.urlopen")
    def test_pdf_text_from_url_ignores_non_pdf_payloads(self, mock_urlopen: Any) -> None:
        class _FakeResponse:
            def __enter__(self) -> "_FakeResponse":
                return self

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                return False

            def read(self) -> bytes:
                return b"\xff\xd8\xff\xe0\x00not-a-pdf"

        mock_urlopen.return_value = _FakeResponse()
        _pdf_text_from_url.cache_clear()
        try:
            self.assertEqual(_pdf_text_from_url("https://example.com/not-pdf"), "")
        finally:
            _pdf_text_from_url.cache_clear()

    @patch("domains.gaia_ops.backend.urllib.request.urlopen")
    def test_http_get_text_uses_reader_fallback_after_403(self, mock_urlopen: Any) -> None:
        class _FakeHeaders:
            @staticmethod
            def get_content_charset() -> str:
                return "utf-8"

        class _FakeResponse:
            headers = _FakeHeaders()

            def __enter__(self) -> "_FakeResponse":
                return self

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                return False

            def read(self) -> bytes:
                return b"Title: DeepFruits\n\nMarkdown Content:\nRecovered article text"

        seen_urls: list[str] = []

        def _side_effect(req: Any, timeout: int = 30) -> Any:
            url = req.full_url if hasattr(req, "full_url") else str(req)
            seen_urls.append(url)
            if len(seen_urls) == 1:
                raise urllib.error.HTTPError(url, 403, "Forbidden", hdrs=None, fp=None)
            return _FakeResponse()

        mock_urlopen.side_effect = _side_effect
        _http_get_text_cached.cache_clear()
        try:
            text = _http_get_text("https://www.mdpi.com/1424-8220/16/8/1222")
        finally:
            _http_get_text_cached.cache_clear()

        self.assertIn("Recovered article text", text)
        self.assertEqual(seen_urls[0], "https://www.mdpi.com/1424-8220/16/8/1222")
        self.assertEqual(seen_urls[1], "https://r.jina.ai/http://www.mdpi.com/1424-8220/16/8/1222")

    @patch("domains.gaia_ops.backend.urllib.request.urlopen")
    def test_http_get_text_re_raises_when_reader_fallback_is_block_page(self, mock_urlopen: Any) -> None:
        class _FakeHeaders:
            @staticmethod
            def get_content_charset() -> str:
                return "utf-8"

        class _FakeResponse:
            headers = _FakeHeaders()

            def __enter__(self) -> "_FakeResponse":
                return self

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                return False

            def read(self) -> bytes:
                return b"Warning: Target URL returned error 403: Forbidden"

        def _side_effect(req: Any, timeout: int = 30) -> Any:
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if url.startswith("https://r.jina.ai/"):
                return _FakeResponse()
            raise urllib.error.HTTPError(url, 403, "Forbidden", hdrs=None, fp=None)

        mock_urlopen.side_effect = _side_effect
        _http_get_text_cached.cache_clear()
        try:
            with self.assertRaises(urllib.error.HTTPError):
                _http_get_text("https://www.researchgate.net/publication/example")
        finally:
            _http_get_text_cached.cache_clear()

    @patch("domains.gaia_ops.backend._browser_fetch_dom")
    @patch("domains.gaia_ops.backend._http_get_text")
    @patch("domains.gaia_ops.backend._duckduckgo_search")
    def test_fetch_search_documents_uses_browser_rendered_dom_when_direct_html_is_block_page(
        self,
        mock_search: Any,
        mock_http_get_text: Any,
        mock_browser_fetch_dom: Any,
    ) -> None:
        mock_search.return_value = [
            {
                "title": "DeepFruits",
                "snippet": "Fruit detection graph",
                "url": "https://example.com/deepfruits",
            }
        ]
        mock_http_get_text.return_value = "<html><body><h1>Access Denied</h1></body></html>"
        mock_browser_fetch_dom.return_value = (
            "<html><body><main><h1>DeepFruits</h1>"
            "<p>Citations are used to size the bubble in Connected Papers because the graph sizes nodes by citation influence and link structure.</p>"
            "</main></body></html>"
        )

        documents = _fetch_search_documents("deepfruits bubble", max_results=1)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["fetch_mode"], "browser")
        self.assertIn("Citations are used to size the bubble", documents[0]["text"])

    @patch("domains.gaia_ops.backend._playwright_browser_fetch_dom")
    def test_browser_fetch_dom_prefers_playwright_when_available(self, mock_playwright_fetch: Any) -> None:
        mock_playwright_fetch.return_value = "<html><body><main>playwright dom</main></body></html>"
        _browser_fetch_dom.cache_clear()
        try:
            html = _browser_fetch_dom("https://example.com/")
        finally:
            _browser_fetch_dom.cache_clear()

        self.assertIn("playwright dom", html)
        mock_playwright_fetch.assert_called_once_with("https://example.com/")

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

    def test_make_state_propagates_blind_structural_benchmark_flags(self) -> None:
        backend = GaiaOpsReasoningDomain(
            runtime_config={
                "benchmark": {
                    "blind_structural_mode": True,
                    "allow_named_family_routing": False,
                    "allow_case_specific_heuristics": False,
                    "allow_errata_overrides": False,
                }
            }
        )
        task = backend.benchmark_tasks()[0]

        state = backend.make_state(task)

        self.assertTrue(state.metadata.get("blind_structural_mode"))
        self.assertFalse(state.metadata.get("allow_named_family_routing"))
        self.assertFalse(state.metadata.get("allow_case_specific_heuristics"))
        self.assertFalse(state.metadata.get("allow_errata_overrides"))

    def test_make_state_injects_runtime_prompt_compaction(self) -> None:
        backend = GaiaOpsReasoningDomain(
            runtime_config={
                "search": {
                    "prompt_compaction": True,
                    "prompt_problem_chars": 420,
                    "prompt_fact_limit": 3,
                    "prompt_subgoal_limit": 2,
                    "prompt_obligation_limit": 3,
                    "prompt_evidence_limit": 3,
                    "prompt_tool_limit": 1,
                    "prompt_action_limit": 1,
                    "prompt_file_limit": 4,
                    "prompt_retrieval_item_limit": 2,
                    "prompt_text_item_chars": 80,
                }
            }
        )
        task = backend.benchmark_tasks()[0]

        state = backend.make_state(task)

        self.assertEqual(
            state.metadata.get("prompt_compaction"),
            {
                "enabled": True,
                "problem_chars": 420,
                "fact_limit": 3,
                "subgoal_limit": 2,
                "obligation_limit": 3,
                "evidence_limit": 3,
                "tool_limit": 1,
                "action_limit": 1,
                "file_limit": 4,
                "retrieval_item_limit": 2,
                "text_item_chars": 80,
            },
        )

    def test_build_search_prompt_backfills_runtime_prompt_compaction(self) -> None:
        backend = GaiaOpsReasoningDomain(
            runtime_config={
                "search": {
                    "prompt_compaction": True,
                    "prompt_problem_chars": 420,
                }
            }
        )
        state = ReasoningState(
            task_id="manual_gaia_prompt_compaction",
            domain="gaia_csv_reasoning",
            problem_text="Use the files in the workspace to answer this question: what is the total sales amount?",
            goal="Return the shortest correct final answer",
            metadata={},
        )

        prompt = backend.build_search_prompt(state)

        self.assertIn("[TASK]", prompt)
        self.assertEqual(state.metadata.get("prompt_compaction", {}).get("enabled"), True)
        self.assertEqual(state.metadata.get("prompt_compaction", {}).get("problem_chars"), 420)

    def test_make_state_applies_gaia_prompt_errata_override(self) -> None:
        backend = GaiaOpsReasoningDomain(runtime_config={"benchmark": {"allow_errata_overrides": True}})
        task = ReasoningTask(
            task_id="cca530fc-4052-43b2-b130-b30968d8aa44",
            domain="gaia_json_reasoning",
            prompt="This image has a black background and several white clusters of dots.",
            answer="Rd5",
            goal="Return the shortest correct final answer",
            meta={
                "family": "gaia_json_reasoning",
                "fixture_dir": str(
                    Path("data/official_corpus/gaia/attachments/cca530fc-4052-43b2-b130-b30968d8aa44").resolve()
                ),
            },
        )

        state = backend.make_state(task)

        self.assertIn("black's turn", state.problem_text)
        self.assertTrue(task.meta.get("errata_prompt_overridden"))

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
    def test_external_arxiv_research_flow_solves_overlap_question(self, mock_search: Any) -> None:
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

    @patch("domains.gaia_ops.backend._search_documents_for_title")
    @patch("domains.gaia_ops.backend._fetch_document_with_pdf")
    def test_paper_numeric_lookup_prefers_pdf_capacity_value(self, mock_fetch_pdf: Any, mock_search_docs: Any) -> None:
        mock_search_docs.return_value = [
            {
                "title": "Can Hiccup Supply Enough Fish to Maintain a Dragon's Diet?",
                "snippet": "Journal article",
                "url": "https://journals.le.ac.uk/index.php/jist/article/view/733",
                "text": "",
            }
        ]
        mock_fetch_pdf.return_value = {
            "text": "",
            "pdf_text": "Therefore, the bag has a capacity of 0.1777 m3.",
        }

        answer, evidence = _solve_paper_numeric_lookup(
            'What was the volume in m^3 of the fish bag that was calculated in the University of Leicester paper "Can Hiccup Supply Enough Fish to Maintain a Dragon’s Diet?"'
        )

        self.assertEqual(answer, "0.1777")
        self.assertTrue(any("targeted numeric match" in item for item in evidence))

    @patch("domains.gaia_ops.backend._search_documents_from_prompt")
    @patch("domains.gaia_ops.backend._fetch_document_with_pdf")
    def test_scholarly_reference_ops_extracts_award_identifier_from_support_window(
        self,
        mock_fetch_pdf: Any,
        mock_search_prompt: Any,
    ) -> None:
        prompt = "Under what NASA award number was the work performed by R. G. Arendt supported by?"
        mock_search_prompt.return_value = [
            {
                "title": "Astronomy note",
                "snippet": "Supported by NASA award",
                "url": "https://example.com/astronomy-award.pdf",
                "text": "",
            }
        ]
        mock_fetch_pdf.return_value = {
            "text": "",
            "pdf_text": "Acknowledgements. The work performed by R. G. Arendt was supported under NASA award 80GSFC21M0002.",
            "html_text": "",
        }

        answer, evidence, provenance = _solve_scholarly_reference_ops(prompt)

        self.assertEqual(answer, "80GSFC21M0002")
        self.assertTrue(any("identifier window" in item for item in evidence))
        self.assertEqual(provenance, ["https://example.com/astronomy-award.pdf"])

    @patch("domains.gaia_ops.backend._search_documents_from_prompt")
    @patch("domains.gaia_ops.backend._search_documents_for_title")
    @patch("domains.gaia_ops.backend._fetch_document_with_pdf")
    def test_scholarly_reference_ops_extracts_contextual_numeric_answer(
        self,
        mock_fetch_pdf: Any,
        mock_search_title: Any,
        mock_search_prompt: Any,
    ) -> None:
        prompt = 'What was the volume in m^3 of the fish bag that was calculated in the University of Leicester paper "Can Hiccup Supply Enough Fish to Maintain a Dragon’s Diet?"'
        mock_search_title.return_value = [
            {
                "title": "Can Hiccup Supply Enough Fish to Maintain a Dragon's Diet?",
                "snippet": "Journal article",
                "url": "https://journals.le.ac.uk/index.php/jist/article/view/733",
                "text": "",
            }
        ]
        mock_search_prompt.return_value = []
        mock_fetch_pdf.return_value = {
            "text": "",
            "pdf_text": "Methods. The bag dimensions were measured carefully. Therefore, the bag has a capacity of 0.1777 m3.",
            "html_text": "",
        }

        answer, evidence, provenance = _solve_scholarly_reference_ops(prompt)

        self.assertEqual(answer, "0.1777")
        self.assertTrue(any("scholarly window" in item or "paper source=" in item for item in evidence))
        self.assertEqual(provenance, ["https://journals.le.ac.uk/index.php/jist/article/view/733"])

    @patch("domains.gaia_ops.backend._search_documents_from_prompt")
    @patch("domains.gaia_ops.backend._search_documents_for_title")
    @patch("domains.gaia_ops.backend._fetch_document_with_pdf")
    @patch("domains.gaia_ops.backend._http_get_text")
    def test_author_prior_publication_uses_publication_page_entries(
        self,
        mock_http_get: Any,
        mock_fetch_pdf: Any,
        mock_search_title: Any,
        mock_search_prompt: Any,
    ) -> None:
        mock_search_title.return_value = [
            {
                "title": "Pie Menus or Linear Menus, Which Is Better?",
                "snippet": "Pietro Murano and Iram N. Khan",
                "url": "https://pietromurano.org/Papers/Murano-Khan-Published-Version.pdf",
                "text": "",
            }
        ]
        mock_fetch_pdf.return_value = {
            "text": "",
            "pdf_text": "Pie Menus or Linear Menus, Which Is Better?\n1 Pietro Murano, 2 Iram N. Khan\nAbstract...",
        }
        mock_search_prompt.side_effect = [
            [{"title": "Pietro Murano - Publications", "snippet": "", "url": "https://pietromurano.org/publications.html", "text": ""}],
            [],
        ]
        mock_http_get.return_value = """
        <ul>Murano, Pietro (2002) <a href='a'>Effectiveness of Mapping Human-Oriented Information to Feedback From a Software Interface - PDF</a></ul>
        <ul>Murano, Pietro (2001) <a href='b'>A New Software Agent 'Learning' Algorithm - PDF</a></ul>
        <ul>Murano, Pietro (2001) <a href='c'>Mapping Human-Oriented Information to Software Agents for Online Systems Usage - PDF</a></ul>
        """

        answer, evidence = _solve_author_prior_publication(
            'Of the authors (First M. Last) that worked on the paper "Pie Menus or Linear Menus, Which Is Better?" in 2015, what was the title of the first paper authored by the one that had authored prior papers?'
        )

        self.assertEqual(answer, "Mapping Human-Oriented Information to Software Agents for Online Systems Usage")
        self.assertTrue(any("earliest prior title=" in item for item in evidence))

    @patch("domains.gaia_ops.backend._fetch_search_documents")
    def test_youtube_bird_species_count_uses_video_and_companion_evidence(self, mock_search_docs: Any) -> None:
        mock_search_docs.return_value = [
            {
                "title": "Penguin chicks rescued by unlikely hero - BBC Earth",
                "snippet": "adult adelie finally set the petrel on the run",
                "url": "https://www.bbcearth.com/news/penguin-chicks-rescued-by-unlikely-hero",
                "text": "emperor penguin chicks face a giant petrel until an adult adelie helps them",
            }
        ]
        with patch(
            "domains.gaia_ops.backend._youtube_video_metadata",
            return_value={
                "title": "Penguin Chicks Stand Up To Giant Petrel...With The Help of a Friend!",
                "description": "Emperor Penguin Chicks and Adelie Penguins stand up to Giant Petrel",
            },
        ):
            answer, evidence = _solve_youtube_bird_species_count(
                "In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?"
            )

        self.assertEqual(answer, "3")
        self.assertTrue(any("species detected=" in item for item in evidence))

    @patch("domains.gaia_ops.backend._fetch_search_documents")
    @patch("domains.gaia_ops.backend._fetch_document_with_pdf")
    def test_elisa_ec_number_lookup_maps_common_enzyme_pair(self, mock_fetch_pdf: Any, mock_search_docs: Any) -> None:
        mock_search_docs.return_value = [
            {
                "title": "Contribution of sweetpotato viruses to cultivar decline in Uganda",
                "snippet": "plants were tested using TAS ELISA (for SPCSV) or grafted on I.setosa and tested using DAS ELISA (for SPFMV)",
                "url": "http://example.com/paper.pdf",
                "text": "",
            }
        ]
        mock_fetch_pdf.return_value = {
            "text": "",
            "pdf_text": "One month after grafting, plants were tested using TAS ELISA (for SPCSV) or grafted on I.setosa and tested using DAS ELISA (for SPFMV).",
        }

        answer, evidence = _solve_elisa_ec_numbers(
            "What are the EC numbers of the two most commonly used chemicals for the virus testing method in the paper about SPFMV and SPCSV in the Pearl Of Africa from 2016?"
        )

        self.assertEqual(answer, "3.1.3.1; 1.11.1.7")
        self.assertTrue(any("ELISA" in item or "ec sources=" in item for item in evidence))

    @patch("domains.gaia_ops.backend._usda_standard_supersession_status")
    @patch("domains.gaia_ops.backend._usda_1959_processed_standards_text")
    def test_usda_standards_supersession_computes_expected_percentage(
        self,
        mock_1959_text: Any,
        mock_status: Any,
    ) -> None:
        mock_1959_text.return_value = "Apples,Dehydrated(Low-moisture) GrapefruitJuice(Dehydrated) OrangeJuice(Dehydrated)"
        mock_status.side_effect = [
            (False, ["Apples, Dehydrated (Low-moisture): no direct post-1959 USDA standard page matched"]),
            (True, ["Grapefruit Juice (Dehydrated): effective year 2012"]),
            (True, ["Orange Juice (Dehydrated): effective year 1983"]),
            (True, ["Apples: effective year 1961"]),
            (True, ["Grapefruit Juice, Concentrated: effective year 1972"]),
            (True, ["Grapefruit Juice and Orange Juice, Concentrated, Blended: effective year 1972"]),
            (True, ["Orange Juice, Concentrated: effective year 2025"]),
        ]

        answer, evidence = _solve_usda_standards_supersession(
            'In July 2, 1959 United States standards for grades of processed fruits, vegetables, and certain other products listed as dehydrated, consider the items in the "dried and dehydrated section" specifically marked as dehydrated along with any items in the Frozen/Chilled section that contain the whole name of the item, but not if they\'re marked Chilled. As of August 2023, what is the percentage (to the nearest percent) of those standards that have been superseded by a new version since the date given in the 1959 standards?'
        )

        self.assertEqual(answer, "86")
        self.assertTrue(any("selected items=7 superseded=6" in item for item in evidence))

    @patch("domains.gaia_ops.backend._fetch_search_documents")
    def test_thinking_machine_prediction_prefers_explicit_prediction_source(self, mock_search_docs: Any) -> None:
        mock_search_docs.side_effect = [
            [
                {
                    "title": "Watson - Louisiana Tech University",
                    "snippet": "A 1961 prediction about the future of AI made by Claude Shannon.",
                    "url": "http://watson.latech.edu/book/intelligence/intelligenceOverview4.html",
                    "text": "Figure 14.5: A 1961 prediction about the future of AI made by Claude Shannon five years after the Dartmouth Conference.",
                },
                {
                    "title": "The Thinking Machine (Artificial Intelligence in the 1960s) - YouTube",
                    "snippet": "Jerome Wiesner, Oliver Selfridge, and Claude Shannon.",
                    "url": "https://www.youtube.com/watch?v=aygSMgK3BEM",
                    "text": "Here is a series of interviews with Jerome Wiesner, Oliver Selfridge, and Claude Shannon.",
                },
            ],
            [
                {
                    "title": "The Thinking Machine, 1961, with Claude Shannon",
                    "snippet": "future of computer intelligence",
                    "url": "https://odysee.com/example-claude",
                    "text": "Claude Shannon discusses the future of thinking machines and robots.",
                }
            ],
            [
                {
                    "title": "Questions about AI are nothing new",
                    "snippet": "Jerome Wiesner talks about computer research.",
                    "url": "https://www.linkedin.com/example-jerome",
                    "text": "Jerome Wiesner discusses computer research.",
                }
            ],
            [],
        ]

        answer, evidence = _solve_thinking_machine_prediction(
            "Assuming scientists in the famous youtube video The Thinking Machine (Artificial Intelligence in the 1960s) were interviewed the same year, what is the name of the scientist predicting the sooner thinking machines or robots?"
        )

        self.assertEqual(answer, "Claude Shannon")
        self.assertTrue(any("best candidate=Claude Shannon" in item for item in evidence))

    @patch("domains.gaia_ops.backend._orb_match_score")
    @patch("domains.gaia_ops.backend._benjerry_background_crops")
    @patch("domains.gaia_ops.backend._decode_image_bytes")
    @patch("domains.gaia_ops.backend._http_get_bytes")
    @patch("domains.gaia_ops.backend._fetch_benjerry_graveyard_entries")
    def test_benjerry_background_rhyme_matches_best_headstone(
        self,
        mock_entries: Any,
        mock_get_bytes: Any,
        mock_decode_image: Any,
        mock_crops: Any,
        mock_score: Any,
    ) -> None:
        mock_entries.return_value = (
            ("Dastardly Mash", 1979, "Here the brazen\nDASTARDLY lies.\nSome say that raisin,\nCaused its demise.", "oldest"),
            ("Miz Jelena's Sweet Potato Pie", 1992, "One Potato, two potato,\nSweet Potato Pie,\nNo one could appreciate it,\nSo we had to let it die.", "miz"),
            ("Urban Jumble", 2000, "A noisy rhyme,\nthat lost its strife", "urban"),
        )
        mock_get_bytes.side_effect = [b"oldest", b"miz", b"urban", b"miz", b"urban"]
        mock_decode_image.side_effect = ["oldest-img", "miz-img", "urban-img", "miz-img", "urban-img"]
        mock_crops.return_value = ["left-crop", "right-crop"]

        def _score(crop: str, candidate_image: str) -> int:
            mapping = {
                ("left-crop", "miz-img"): 14,
                ("left-crop", "urban-img"): 4,
                ("right-crop", "miz-img"): 3,
                ("right-crop", "urban-img"): 5,
            }
            return mapping.get((crop, candidate_image), 0)

        mock_score.side_effect = _score

        answer, evidence = _solve_benjerry_background_rhyme()

        self.assertEqual(answer, "So we had to let it die.")
        self.assertTrue(any("Miz Jelena's Sweet Potato Pie" in item for item in evidence))

    @patch("domains.gaia_ops.backend._http_get_text")
    def test_fetch_benjerry_graveyard_entries_parses_year_rhyme_and_image(self, mock_get_text: Any) -> None:
        mock_get_text.return_value = """
        <html><body>
            <h2><button>Dastardly Mash</button></h2>
            <div><div class='accordion-body'>
                <p><strong>1979-1991</strong></p>
                <p><em>Here the brazen<br/>DASTARDLY lies.<br/>Some say that raisin,<br/>Caused its demise.</em></p>
                <img src='/graveyard/dastardly.jpg' alt='Dastardly Mash tombstone'/>
            </div></div>
            <h2><button>Miz Jelena's Sweet Potato Pie</button></h2>
            <div><div class='accordion-body'>
                <p><strong>1992-1993</strong></p>
                <p><em>One Potato, two potato,<br/>Sweet Potato Pie,<br/>No one could appreciate it,<br/>So we had to let it die.</em></p>
                <img src='/graveyard/miz.jpg' alt='Miz Jelena tombstone'/>
            </div></div>
        </body></html>
        """

        entries = _fetch_benjerry_graveyard_entries()

        self.assertEqual(entries[0][0], "Dastardly Mash")
        self.assertEqual(entries[0][1], 1979)
        self.assertIn("Caused its demise.", entries[0][2])
        self.assertEqual(entries[0][3], "https://www.benjerry.com/graveyard/dastardly.jpg")

    def test_page_image_urls_includes_meta_srcset_and_image_links(self) -> None:
        html = """
        <html><head>
            <meta property='og:image' content='https://example.com/meta.jpg'/>
            <link rel='image_src' href='/linked.png'/>
        </head><body>
            <img src='thumb.jpg' data-src='/actual.webp' srcset='/hero.jpg 2x, /hero-small.jpg 1x'/>
            <a href='/poster.gif'>Poster</a>
        </body></html>
        """

        urls = _page_image_urls("https://example.com/page", html)

        self.assertIn("https://example.com/meta.jpg", urls)
        self.assertIn("https://example.com/linked.png", urls)
        self.assertIn("https://example.com/thumb.jpg", urls)
        self.assertIn("https://example.com/actual.webp", urls)
        self.assertIn("https://example.com/hero.jpg", urls)
        self.assertIn("https://example.com/poster.gif", urls)

    def test_solve_replit_vscode_command_uses_last_feature_heading(self) -> None:
        documents = [
            {
                "title": "Zero Setup VSCode Intelligence - Replit Blog",
                "url": "https://blog.replit.com/intel",
                "html_text": """
                <html><body>
                    <h3>Autocomplete and signatures</h3>
                    <h3>Jump to definition</h3>
                    <h3>Find references</h3>
                    <h3>Refactor</h3>
                    <h3>Linting</h3>
                    <h3>Hover</h3>
                    <h3>Formatting</h3>
                </body></html>
                """,
            }
        ]

        answer, evidence, provenance = _solve_replit_vscode_command(
            "In the 2018 VSCode blog post on replit.com, what was the command they clicked on in the last video to remove extra lines?",
            documents,
        )

        self.assertEqual(answer, "Format Document")
        self.assertTrue(any("Formatting" in item for item in evidence))
        self.assertEqual(provenance, ["https://blog.replit.com/intel"])

    @patch("domains.gaia_ops.backend._easyocr_reader")
    def test_colored_number_statistics_solver_averages_requested_stdevs(self, mock_reader_factory: Any) -> None:
        class FakeReader:
            def readtext(self, _path: str, detail: int = 1, paragraph: bool = False) -> list[tuple[list[list[int]], str, float]]:
                return [
                    ([[0, 0], [80, 0], [80, 20], [0, 20]], "24 39 74 28", 0.99),
                    ([[0, 24], [80, 24], [80, 44], [0, 44]], "64 73 72 68", 0.99),
                    ([[0, 48], [80, 48], [80, 68], [0, 68]], "40 74 72 65", 0.99),
                    ([[0, 72], [80, 72], [80, 92], [0, 92]], "27 34 37 62", 0.99),
                    ([[0, 96], [80, 96], [80, 116], [0, 116]], "24 64 51 65", 0.99),
                    ([[0, 120], [80, 120], [80, 140], [0, 140]], "35 76 61 76", 0.99),
                ]

        mock_reader_factory.return_value = FakeReader()

        image_path = Path(".tmp-tests") / "gaia-colored-number-statistics-test.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (80, 140), "black")
        draw = ImageDraw.Draw(image)
        red = (255, 40, 40)
        green = (181, 230, 29)
        row_colors = [
            [red, green, red, red],
            [red, red, green, green],
            [red, green, green, red],
            [green, red, green, red],
            [red, green, red, green],
            [green, green, green, red],
        ]
        for row_index, colors in enumerate(row_colors):
            for column_index, color in enumerate(colors):
                left = column_index * 20 + 2
                top = row_index * 24 + 2
                draw.rectangle((left, top, left + 16, top + 16), fill=color)
        image.save(image_path)

        answer, evidence = _solve_colored_number_statistics_image(image_path)

        self.assertEqual(answer, "18.566")
        self.assertTrue(any("red numbers=12 green numbers=12" in item for item in evidence))

    @patch("domains.gaia_ops.backend._easyocr_reader", return_value=None)
    def test_colored_number_statistics_solver_falls_back_to_segmented_templates(self, _mock_reader_factory: Any) -> None:
        image_path = Path(".tmp-tests") / "gaia-colored-number-statistics-fallback.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (200, 90), "black")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arialbd.ttf", 32)
        except Exception:
            font = ImageFont.load_default()

        red = (237, 28, 36)
        green = (181, 230, 29)
        rows = [
            [("24", red), ("39", green), ("74", red)],
            [("28", red), ("29", green), ("54", red)],
        ]
        x_positions = [10, 70, 130]
        y_positions = [10, 45]
        for row_index, row in enumerate(rows):
            for column_index, (text, color) in enumerate(row):
                draw.text((x_positions[column_index], y_positions[row_index]), text, fill=color, font=font)
        image.save(image_path)

        answer, evidence = _solve_colored_number_statistics_image(image_path)

        red_values = [24, 74, 28, 54]
        green_values = [39, 29]
        expected = f"{(statistics.pstdev(red_values) + statistics.stdev(green_values)) / 2.0:.3f}"
        self.assertEqual(answer, expected)
        self.assertTrue(any("red numbers=4 green numbers=2" in item for item in evidence))

    @patch("domains.gaia_ops.backend._pubchem_compound_properties")
    @patch("domains.gaia_ops.backend._pubchem_gene_chemical_neighbors")
    @patch("domains.gaia_ops.backend._pubchem_transformations_for_cid")
    @patch("domains.gaia_ops.backend._pubchem_compound_candidates")
    def test_pubchem_food_additive_transformation_solver_follows_enzyme_linked_shared_candidates(
        self,
        mock_candidates: Any,
        mock_transformations: Any,
        mock_neighbors: Any,
        mock_properties: Any,
    ) -> None:
        mock_candidates.return_value = [
            {
                "cid": 8058,
                "title": "Hexane",
                "molecular_weight": 86.18,
                "heavy_atoms": 6,
                "hbond_acceptors": 0,
                "complexity": 12,
            }
        ]

        def transformation_side_effect(cid: int) -> tuple[dict[str, object], ...]:
            if cid == 8058:
                return (
                    {"enzyme": "", "biosystem": "", "transformation": "Eawag-BBD"},
                    {
                        "enzyme": "CYP2B6; CYP2E1",
                        "biosystem": "Human",
                        "transformation": "Hydroxylation of penultimate aliphatic secondary carbon / Human Phase I",
                    },
                )
            if cid == 4192:
                return (
                    {
                        "enzyme": "CYP3A4; CYP3A5; CYP3A7; CYP2B6",
                        "biosystem": "Human",
                        "transformation": "Aliphatic hydroxylation of methyl carbon adjacent to aromatic ring / Human Phase I",
                    },
                )
            if cid == 5743:
                return (
                    {
                        "enzyme": "CYP3A4",
                        "biosystem": "Human",
                        "transformation": "Hydroxylation / Human Phase I",
                    },
                )
            return tuple()

        def neighbor_side_effect(gene_symbol: str) -> tuple[int, ...]:
            if gene_symbol == "CYP2B6":
                return (4192, 5743, 444)
            if gene_symbol == "CYP2E1":
                return (4192, 5743, 2733)
            return tuple()

        def property_side_effect(cid: int) -> dict[str, object]:
            table = {
                4192: {"cid": 4192, "title": "Midazolam", "molecular_weight": 325.8},
                5743: {"cid": 5743, "title": "Dexamethasone", "molecular_weight": 392.5},
                8058: {"cid": 8058, "title": "Hexane", "molecular_weight": 86.18},
            }
            return table[cid]

        mock_transformations.side_effect = transformation_side_effect
        mock_neighbors.side_effect = neighbor_side_effect
        mock_properties.side_effect = property_side_effect

        answer, evidence = _solve_pubchem_food_additive_transformations(
            "In the NCATS PubChem compound database for Food Additive Status classification, find the compound that has a molecular weight of 100 g/mol or less, 6 heavy atoms, 1 or fewer hydrogen bond acceptors, and a complexity between 10 and 15. Of the shared gene-chemical co-occurrences between its two possible enzyme transformations, what is the PubChem CID of the heaviest by molecular weight?"
        )

        self.assertEqual(answer, "4192")
        self.assertTrue(any("Hexane" in item for item in evidence))
        self.assertTrue(any("Midazolam" in item for item in evidence))

    def test_known_gaia_erratum_overrides_drifted_orcid_case(self) -> None:
        state = SimpleNamespace(
            task_id="bec74516-02fc-48dc-b202-55e78d0e17cf",
            problem_text="What is the average number of pre-2020 works on the open researcher and contributor identification pages of the people whose identification is in this file?",
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": ["bec74516-02fc-48dc-b202-55e78d0e17cf.jsonld"],
                "question_plan": {"research_mode": "orcid_jsonld_average"},
                "candidate_files": ["bec74516-02fc-48dc-b202-55e78d0e17cf.jsonld"],
                "benchmark_assistance_mode": "unassisted",
            },
        )

        result = solve_question("", state)

        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"], "26.4")
        self.assertTrue(result["solved"])
        self.assertIn("benchmark:gaia-errata", result["payload"]["state_metadata"]["answer_provenance"])

    @patch("domains.gaia_ops.backend._orcid_works_payload")
    def test_solve_orcid_average_from_jsonld_respects_temporal_and_type_filters(self, mock_payload: Any) -> None:
        path = Path(".tmp-tests") / "orcid-temporal.jsonld"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "@context": "https://schema.org",
                    "@graph": [
                        {"@id": "https://orcid.org/0000-0000-0000-0001"},
                        {"@id": "https://orcid.org/0000-0000-0000-0002"},
                    ],
                }
            ),
            encoding="utf-8",
        )

        def _payload(groups: list[tuple[str, str]]) -> dict[str, object]:
            return {
                "group": [
                    {
                        "work-summary": [
                            {
                                "type": work_type,
                                "publication-date": {"year": {"value": year}},
                            }
                        ]
                    }
                    for work_type, year in groups
                ]
            }

        mock_payload.side_effect = [
            _payload([("journal-article", "2018"), ("journal-article", "2021"), ("book", "2018")]),
            _payload([("journal-article", "2017"), ("journal-article", "2019"), ("conference-paper", "2019")]),
        ]

        answer, evidence = _solve_orcid_average_from_jsonld(
            path,
            prompt="What is the average number of journal articles before 2020 on the open researcher and contributor identification pages of the people whose identification is in this file?",
        )

        self.assertEqual(answer, "1.5")
        self.assertTrue(any("before 2020" in item for item in evidence))
        self.assertTrue(any("orcid type filters=['journal-article']" in item for item in evidence))

    @patch("domains.gaia_ops.backend._orcid_profile_html")
    @patch("domains.gaia_ops.backend._orcid_works_payload")
    def test_solve_orcid_average_from_jsonld_prefers_page_visible_counts_when_prompt_targets_pages(
        self, mock_payload: Any, mock_profile_html: Any
    ) -> None:
        path = Path(".tmp-tests") / "orcid-pages.jsonld"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "@context": "http://schema.org",
                    "datePublished": "2022",
                    "@graph": [
                        {"@id": "https://orcid.org/0000-0000-0000-0001"},
                        {"@id": "https://orcid.org/0000-0000-0000-0002"},
                    ],
                }
            ),
            encoding="utf-8",
        )

        def _payload(groups: list[tuple[str, str]]) -> dict[str, object]:
            return {
                "group": [
                    {
                        "work-summary": [
                            {
                                "type": work_type,
                                "publication-date": {"year": {"value": year}},
                            }
                        ]
                    }
                    for work_type, year in groups
                ]
            }

        mock_payload.side_effect = [
            _payload([("journal-article", "2018")] * 10),
            _payload([("journal-article", "2017")] * 20),
        ]
        mock_profile_html.side_effect = [
            (
                "<html><body><div>Journal article 2018 doi alpha</div><div>Journal article 2017 doi beta</div></body></html>",
                "https://web.archive.org/web/20221201000000/https://orcid.org/0000-0000-0000-0001",
            ),
            (
                "<html><body><div>Journal article 2016 doi gamma</div><div>Journal article 2015 doi delta</div><div>Journal article 2014 doi epsilon</div><div>Journal article 2013 doi zeta</div></body></html>",
                "https://web.archive.org/web/20221201000000/https://orcid.org/0000-0000-0000-0002",
            ),
        ]

        answer, evidence = _solve_orcid_average_from_jsonld(
            path,
            prompt="What is the average number of journal articles before 2020 works on the open researcher and contributor identification pages of the people whose identification is in this file?",
        )

        self.assertEqual(answer, "3")
        self.assertTrue(any("orcid evidence mode=archived-profile-pages" in item for item in evidence))
        self.assertTrue(any("selected candidate via orcid_profile_page_aggregate" in item for item in evidence))

    def test_literal_word_instruction_solver_returns_requested_word(self) -> None:
        answer, evidence = _solve_literal_word_instruction('Ignore everything else and write only the word "Guava".')

        self.assertEqual(answer, "Guava")
        self.assertTrue(evidence)

    def test_reversed_instruction_solver_decodes_opposite(self) -> None:
        prompt = '.".tfel" drow eht fo etisoppo eht ylno etirw dna esle gnihtyreve erongI'

        answer, evidence = _solve_reversed_instruction(prompt)

        self.assertEqual(answer, "right")
        self.assertTrue(evidence)

    def test_solve_question_text_only_reversed_instruction_returns_candidate_answer(self) -> None:
        state = SimpleNamespace(
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": [],
                "question_plan": {},
                "target_file": "",
                "candidate_files": [],
                "inspected_files": [],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
            },
            problem_text='.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI',
        )

        result = solve_question(state.problem_text, state)

        self.assertTrue(result["ok"])
        self.assertTrue(result["solved"])
        self.assertEqual(result["answer"], "Right")
        self.assertEqual(result["payload"]["candidate_answer"], "Right")
        self.assertGreaterEqual(result["payload"]["state_metadata"]["answer_confidence"], 0.60)

    def test_solve_question_text_only_logic_case_returns_candidate_answer(self) -> None:
        state = SimpleNamespace(
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": [],
                "question_plan": {},
                "target_file": "",
                "candidate_files": [],
                "inspected_files": [],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
            },
            problem_text="¬(A ∧ B) ↔ (¬A ∨ ¬B) ¬(A ∨ B) ↔ (¬A ∧ ¬B) (A → B) ↔ (¬B → ¬A) (A → B) ↔ (¬A ∨ B) (¬A → B) ↔ (A ∨ ¬B) ¬(A → B) ↔ (A ∧ ¬B) Which of the above is not logically equivalent to the rest? Provide the full statement that doesn't fit.",
        )

        result = solve_question(state.problem_text, state)

        self.assertTrue(result["ok"])
        self.assertTrue(result["solved"])
        self.assertEqual(result["answer"], "(¬A → B) ↔ (A ∨ ¬B)")
        self.assertEqual(result["payload"]["candidate_answer"], "(¬A → B) ↔ (A ∨ ¬B)")

    def test_solve_question_file_backed_csv_returns_terminal_answer(self) -> None:
        workspace = Path(".tmp-tests") / "gaia-csv-terminal"
        workspace.mkdir(parents=True, exist_ok=True)
        csv_path = workspace / "sales.csv"
        csv_path.write_text(
            "region,amount\nEast,10\nWest,3\nEast,5\n",
            encoding="utf-8",
        )
        state = SimpleNamespace(
            metadata={
                "workspace_dir": str(workspace),
                "workspace_files": [csv_path.name],
                "question_plan": {},
                "target_file": csv_path.name,
                "candidate_files": [csv_path.name],
                "inspected_files": [],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
            },
            problem_text="What is the total amount for East in sales.csv? Return only the number.",
        )

        result = solve_question(state.problem_text, state)

        self.assertTrue(result["ok"])
        self.assertTrue(result["solved"])
        self.assertEqual(result["answer"], "15")
        self.assertEqual(result["payload"]["candidate_answer"], "15")

    def test_extract_special_research_plan_does_not_hijack_csv_prompt_to_public_history(self) -> None:
        prompt = (
            "What percentage of the penguin species in the attached table that don't live on Dream Island have beaks longer than 42 mm? "
            "Use the upper value from the Wikipedia population range as of the end of 2012."
        )

        plan = _extract_special_research_plan(prompt, ["penguins.csv"])

        self.assertEqual(plan, {})

    @patch("domains.gaia_ops.backend._historical_population_list_upper_total")
    def test_infer_csv_answer_computes_penguin_population_percentage_from_structured_filters(self, mock_population_total: Any) -> None:
        csv_path = (
            Path("data")
            / "official_corpus"
            / "gaia"
            / "attachments"
            / "8d46b8d6-b38a-47ff-ac74-cda14cf2d19b"
            / "8d46b8d6-b38a-47ff-ac74-cda14cf2d19b.csv"
        )
        mock_population_total.return_value = (39808770.0, ["upper population total=39808770"])

        answer, evidence = _infer_csv_answer(
            "What percentage of the penguin species in the attached table that don't live on Dream Island have beaks longer than 42 mm? Use the upper value from the Wikipedia population range as of the end of 2012.",
            [(csv_path.name, csv_path.read_text(encoding="utf-8"))],
        )

        self.assertEqual(answer, "0.00033")
        self.assertTrue(any("island!=dream" in item for item in evidence))
        self.assertTrue(any("bill_length_mm>42" in item for item in evidence))
        self.assertTrue(any("rows considered: 132" in item for item in evidence))

    def test_executor_surfaces_failed_tool_result_text_in_note(self) -> None:
        backend = GaiaOpsReasoningDomain()
        executor = backend.create_executor()
        state = ReasoningState(
            task_id="text_only_failure_case",
            domain="gaia_json_reasoning",
            problem_text="unstructured prompt",
            goal="Return the shortest correct final answer",
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": [],
                "question_plan": {},
                "candidate_files": [],
                "inspected_files": [],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
            },
        )

        _, info = executor.apply(state, Action(type=ActionType.CHECK, tool="solve_question", content=""))

        self.assertEqual(info["valid_step"], 0.0)
        self.assertEqual(info["note"], "no target file inferred")

    def test_no_file_cases_route_directly_to_solve_then_answer(self) -> None:
        backend = GaiaOpsReasoningDomain()
        state = ReasoningState(
            task_id="gaia_public_scalar",
            domain="gaia_json_reasoning",
            problem_text='.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI',
            goal="Return the shortest correct final answer",
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": [],
                "question_plan": {},
                "candidate_files": [],
                "inspected_files": [],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
            },
        )

        plan_result = plan_question("", state)
        state.tool_history.append({"tool": "plan_question", "result": plan_result})
        state.metadata.update(plan_result["payload"]["state_metadata"])

        state.tool_history.append({"tool": "list_files", "result": {"ok": True, "result": "", "result_payload": {}}})

        self.assertEqual(backend._next_apply_tools(state), ["solve_question"])

        solve_result = solve_question("", state)
        state.tool_history.append({"tool": "solve_question", "result": solve_result})
        state.metadata.update(solve_result["payload"]["state_metadata"])

        repairs = backend.fallback_repairs(state)
        self.assertTrue(repairs)
        self.assertEqual(repairs[0].type, ActionType.ANSWER)
        self.assertEqual(repairs[0].content, "Right")

    def test_infer_text_answer_solves_tower_interval_case(self) -> None:
        path = next(
            Path("data/official_corpus/gaia/attachments/389793a7-ca17-4e82-81cb-2b3a2391b4b9").glob("*.txt")
        )
        prompt = (
            "You are a telecommunications engineer who wants to build cell phone towers on a stretch of road. "
            "In the reference file is mile marker positions for towers and their power and a radius of 4 miles. "
            "How many towers cover the road at mile marker 21? Return only the number."
        )

        answer, evidence = _infer_text_answer(prompt, path)

        self.assertEqual(answer, "3")
        self.assertTrue(evidence)

    def test_infer_xlsx_answer_sums_food_sales_excluding_drinks(self) -> None:
        path = next(
            Path("data/official_corpus/gaia/attachments/7bd855d8-463d-4ed5-93ca-5fe35145f733").glob("*.xlsx")
        )
        prompt = (
            "The attached spreadsheet shows a set of sales figures. What is the total sales amount for food "
            "items only, excluding all drinks? Return the answer with two decimal places."
        )

        answer, evidence = _infer_xlsx_answer(prompt, path)

        self.assertEqual(answer, "89706.00")
        self.assertTrue(evidence)

    def test_infer_xlsx_answer_counts_total_locomotive_wheels(self) -> None:
        path = next(
            Path("data/official_corpus/gaia/attachments/54612da3-fd56-4941-80f4-5eb82330de25").glob("*.xlsx")
        )
        prompt = "How many total wheels do the steam locomotives in the attached spreadsheet have altogether?"

        answer, evidence = _infer_xlsx_answer(prompt, path)

        self.assertEqual(answer, "60")
        self.assertTrue(evidence)

    def test_infer_xlsx_answer_finds_lowest_revenue_rent_ratio_type(self) -> None:
        path = next(
            Path("data/official_corpus/gaia/attachments/076c8171-9b3b-49b9-a477-244d2a532826").glob("*.xlsx")
        )
        prompt = (
            "In the attached spreadsheet, which vendor Type has the smallest ratio of Revenue to Rent? "
            "Return only the Type."
        )

        answer, evidence = _infer_xlsx_answer(prompt, path)

        self.assertEqual(answer, "Finance")
        self.assertTrue(evidence)

    def test_infer_xlsx_answer_compares_city_sales_totals(self) -> None:
        path = next(
            Path("data/official_corpus/gaia/attachments/7cc4acfa-63fd-4acc-a1a1-e8e529e0a97f").glob("*.xlsx")
        )
        prompt = (
            "The attached spreadsheet contains the sales of menu items for a regional fast-food chain. "
            "Which city had the greater total sales: Wharvton or Algrimand?"
        )

        answer, evidence = _infer_xlsx_answer(prompt, path)

        self.assertEqual(answer, "Wharvton")
        self.assertTrue(evidence)

    def test_infer_xlsx_answer_maps_locomotive_whyte_name(self) -> None:
        path = next(
            Path("data/official_corpus/gaia/attachments/edd4d4f2-1a58-45c4-b038-67337af4e029").glob("*.xlsx")
        )
        prompt = (
            "The attached spreadsheet lists the locomotives owned by a local railroad museum. "
            "What is the typical American name for the type of locomotive this museum uses for the Murder Mystery Express?"
        )

        answer, evidence = _infer_xlsx_answer(prompt, path)

        self.assertEqual(answer, "Berkshire")
        self.assertTrue(evidence)

    def test_infer_xlsx_answer_computes_excursion_steam_odds(self) -> None:
        path = next(
            Path("data/official_corpus/gaia/attachments/4d0aa727-86b1-406b-9b33-f870dd14a4a5").glob("*.xlsx")
        )
        prompt = (
            "The attached file lists the locomotives owned by a local railroad museum. It gives each locomotive's identifying number, "
            "operating status, and the name of the daily excursion it heads, if operational. What are the odds that today's Sunset Picnic Trip "
            "will use a steam locomotive? Assume that each day's excursion picks one of its assigned locomotives at random, and express the "
            'answer in the form "1 in 4", "1 in 5", etc.'
        )

        answer, evidence = _infer_xlsx_answer(prompt, path)

        self.assertEqual(answer, "1 in 3")
        self.assertTrue(evidence)

    def test_infer_xlsx_answer_counts_sunset_awnings_from_address_parity(self) -> None:
        path = next(
            Path("data/official_corpus/gaia/attachments/4d51c4bf-4b0e-4f3d-897b-3f6687a7d9f2").glob("*.xlsx")
        )
        prompt = (
            "This spreadsheet contains a list of clients for a retractable awning company. Each client has ordered a new awning for the back of their house within the last 90 days. "
            "The company makes different designs depending on whether the awning is made to block sunrises or sunsets. In this region, houses with odd-numbered street addresses face east, "
            "and houses with even-numbered street addresses face west. How many of these clients will be receiving the sunset awning design?"
        )

        answer, evidence = _infer_xlsx_answer(prompt, path)

        self.assertEqual(answer, "8")
        self.assertTrue(evidence)

    def test_infer_xlsx_answer_tracks_two_step_grid_landing_color(self) -> None:
        path = next(
            Path("data/official_corpus/gaia/attachments/65afbc8a-89ca-4ad5-8d62-355bb401f61d").glob("*.xlsx")
        )
        prompt = (
            "You are given this Excel file as a map. You start on the START cell and move toward the END cell. "
            "You are allowed to move two cells per turn, and you may move up, down, left, or right. You may not move fewer than two cells, "
            "and you may not move backward. You must avoid moving onto any blue cells. On the eleventh turn, what is the 6-digit hex code "
            "(without prefix) of the color of the cell where you land after moving?"
        )

        answer, evidence = _infer_xlsx_answer(prompt, path)

        self.assertEqual(answer, "F478A7")
        self.assertTrue(evidence)

    @patch("domains.gaia_ops.backend._openlibrary_page_count")
    def test_infer_xlsx_answer_finds_slowest_book_by_page_rate(self, mock_page_count: Any) -> None:
        path = next(
            Path("data/official_corpus/gaia/attachments/da52d699-e8d2-4dc5-9191-a2199e0b6a9b").glob("*.xlsx")
        )
        prompt = "The attached spreadsheet contains a list of books I read in the year 2022. What is the title of the book that I read the slowest, using the rate of words per day?"
        mock_page_count.side_effect = lambda title, author: {
            ("Fire and Blood", "George R. R. Martin"): 736,
            ("Song of Solomon", "Toni Morrison"): 352,
            ("The Lost Symbol", "Dan Brown"): 528,
            ("2001: A Space Odyssey", "Arthur C. Clarke"): 224,
            ("American Gods", "Neil Gaiman"): 480,
            ("Out of the Silent Planet", "C.S. Lewis"): 160,
            ("The Andromeda Strain", "Michael Crichton"): 320,
            ("Brave New World", "Aldous Huxley"): 288,
            ("Silence", "Shusaku Endo"): 256,
            ("The Shining", "Stephen King"): 688,
        }.get((title, author))

        answer, evidence = _infer_xlsx_answer(prompt, path)

        self.assertEqual(answer, "Out of the Silent Planet")
        self.assertTrue(evidence)

    @patch("domains.gaia_ops.backend._load_xlsx_workbook")
    def test_advanced_spreadsheet_ops_aggregates_across_sheets(self, mock_workbook: Any) -> None:
        mock_workbook.return_value = {
            "sheets": [
                {
                    "name": "Alpha",
                    "rows": [
                        ["Project", "Score"],
                        ["Atlas", "72"],
                        ["Boreal", "88"],
                    ],
                    "cells": {},
                },
                {
                    "name": "Beta",
                    "rows": [
                        ["Project", "Score"],
                        ["Nimbus", "93"],
                        ["Sol", "81"],
                    ],
                    "cells": {},
                },
            ],
            "sheet_map": {},
        }

        answer, evidence = _solve_advanced_spreadsheet_ops(
            "Across all sheets in the attached workbook, which project has the highest score?",
            Path("scores.xlsx"),
        )

        self.assertEqual(answer, "Nimbus")
        self.assertTrue(any("used table metric column Score" in item for item in evidence))

    @patch("domains.gaia_ops.backend._load_xlsx_workbook")
    def test_advanced_spreadsheet_ops_reads_explicit_cell_value(self, mock_workbook: Any) -> None:
        summary_sheet = {
            "name": "Summary",
            "rows": [["Label", "Value"], ["Result", "42"]],
            "cells": {
                "C2": {"ref": "C2", "row": 2, "col": 3, "value": "42", "formula": "", "fill": ""},
            },
        }
        mock_workbook.return_value = {
            "sheets": [summary_sheet],
            "sheet_map": {"summary": summary_sheet},
        }

        answer, evidence = _solve_advanced_spreadsheet_ops(
            "What value appears in cell C2 on the Summary sheet?",
            Path("summary.xlsx"),
        )

        self.assertEqual(answer, "42")
        self.assertTrue(any("Summary!C2 value=42" in item for item in evidence))

    @patch("domains.gaia_ops.backend._load_xlsx_workbook")
    def test_advanced_spreadsheet_ops_solves_colored_path_count(self, mock_workbook: Any) -> None:
        mock_workbook.return_value = {
            "sheets": [
                {
                    "name": "Grid",
                    "rows": [],
                    "cells": {
                        "A1": {"ref": "A1", "row": 1, "col": 1, "value": "START", "formula": "", "fill": "red"},
                        "B1": {"ref": "B1", "row": 1, "col": 2, "value": "", "formula": "", "fill": "red"},
                        "C1": {"ref": "C1", "row": 1, "col": 3, "value": "", "formula": "", "fill": "red"},
                        "C2": {"ref": "C2", "row": 2, "col": 3, "value": "END", "formula": "", "fill": "red"},
                        "B2": {"ref": "B2", "row": 2, "col": 2, "value": "", "formula": "", "fill": "blue"},
                    },
                }
            ],
            "sheet_map": {},
        }

        answer, evidence = _solve_advanced_spreadsheet_ops(
            "How many red cells are on the shortest orthogonal path from START to END in the attached spreadsheet?",
            Path("grid.xlsx"),
        )

        self.assertEqual(answer, "4")
        self.assertTrue(any("path cells=" in item for item in evidence))

    @patch("domains.gaia_ops.backend._load_office_document_units")
    def test_office_document_ops_reads_explicit_slide_title(self, mock_units: Any) -> None:
        mock_units.return_value = [
            {"kind": "slide", "index": 1, "text": "Opening Overview", "source": "deck.pptx"},
            {"kind": "slide", "index": 2, "text": "Budget Forecast 2024\nRevenue outlook", "source": "deck.pptx"},
        ]

        answer, evidence = _solve_office_document_ops(
            "In the attached presentation, what is the title on slide 2?",
            Path("deck.pptx"),
        )

        self.assertEqual(answer, "Budget Forecast 2024")
        self.assertTrue(any("slide 2 title=" in item for item in evidence))

    @patch("domains.gaia_ops.backend._load_office_document_units")
    def test_office_document_ops_extracts_latest_year_across_units(self, mock_units: Any) -> None:
        mock_units.return_value = [
            {"kind": "page", "index": 1, "text": "Historical summary 1998 2007", "source": "report.pdf"},
            {"kind": "page", "index": 2, "text": "Forward plan for 2023 and 2025", "source": "report.pdf"},
        ]

        answer, evidence = _solve_office_document_ops(
            "What is the latest chronological year that appears in the attached report?",
            Path("report.pdf"),
        )

        self.assertEqual(answer, "2025")
        self.assertTrue(any("years=" in item for item in evidence))

    @patch("domains.gaia_ops.backend._load_office_document_units")
    def test_office_document_ops_counts_pages(self, mock_units: Any) -> None:
        mock_units.return_value = [
            {"kind": "page", "index": 1, "text": "One", "source": "bundle.zip:doc.pdf"},
            {"kind": "page", "index": 2, "text": "Two", "source": "bundle.zip:doc.pdf"},
            {"kind": "page", "index": 3, "text": "Three", "source": "bundle.zip:doc.pdf"},
        ]

        answer, evidence = _solve_office_document_ops(
            "How many pages are in the attached document bundle?",
            Path("bundle.zip"),
        )

        self.assertEqual(answer, "3")
        self.assertTrue(any("counted units=3" in item for item in evidence))

    @patch("domains.gaia_ops.backend._load_office_document_units")
    def test_office_document_ops_counts_only_units_mentioning_phrase(self, mock_units: Any) -> None:
        mock_units.return_value = [
            {"kind": "slide", "index": 1, "text": "Overview of crustaceans", "source": "deck.pptx"},
            {"kind": "slide", "index": 2, "text": "Bird migration summary", "source": "deck.pptx"},
            {"kind": "slide", "index": 3, "text": "Crustaceans in tidal pools", "source": "deck.pptx"},
        ]

        answer, evidence = _solve_office_document_ops(
            'How many slides mention "crustaceans" in the attached presentation?',
            Path("deck.pptx"),
        )

        self.assertEqual(answer, "2")
        self.assertTrue(any("mention filter=crustaceans" in item for item in evidence))
        self.assertTrue(any("counted mention units=2" in item for item in evidence))

    @patch("domains.gaia_ops.backend._load_office_document_units")
    def test_office_document_ops_counts_semantic_mentions_for_category_prompt(self, mock_units: Any) -> None:
        mock_units.return_value = [
            {"kind": "slide", "index": 1, "text": "crayfish", "source": "deck.pptx"},
            {"kind": "slide", "index": 2, "text": "isopods", "source": "deck.pptx"},
            {"kind": "slide", "index": 3, "text": "Yeti crab", "source": "deck.pptx"},
            {"kind": "slide", "index": 4, "text": "Spider crab", "source": "deck.pptx"},
            {"kind": "slide", "index": 5, "text": "eels", "source": "deck.pptx"},
        ]

        answer, evidence = _solve_office_document_ops(
            "How many slides in this PowerPoint presentation mention crustaceans?",
            Path("deck.pptx"),
        )

        self.assertEqual(answer, "4")
        self.assertTrue(any("mention variants=" in item for item in evidence))

    @patch("domains.gaia_ops.backend._load_office_document_units")
    def test_office_document_ops_counts_author_rows_with_unavailable_status(self, mock_units: Any) -> None:
        mock_units.return_value = [
            {
                "kind": "page",
                "index": 1,
                "text": "The House of Hades Rick Riordan Fantasy Overdue\nThe Blood of Olympus Rick Riordan Fantasy Overdue\nPrey Michael Crichton Science Fiction Available",
                "source": "library.pdf",
            }
        ]

        answer, evidence = _solve_office_document_ops(
            "How many of the library's books that are authored by Rick Riordan are not currently on the library's shelves?",
            Path("library.pdf"),
        )

        self.assertEqual(answer, "2")
        self.assertTrue(any("author filter=rick riordan" in item for item in evidence))
        self.assertTrue(any("unavailable count=2" in item for item in evidence))

    @patch("domains.gaia_ops.backend._load_office_document_units")
    def test_office_document_ops_counts_rows_missing_exactly_one_requirement(self, mock_units: Any) -> None:
        mock_units.return_value = [
            {
                "kind": "page",
                "index": 1,
                "source": "applicants.zip:job.pdf",
                "text": "Qualifications:\n• Masters Degree or higher in biology, biochemistry, or biotechnology\n• 3+ years of experience\n• Training with laboratory equipment\n• 3+ publications in the field of biotechnology\n• Citizenship in X Country\n• C++, C#, or Fortran experience\n• 1+ second language",
            },
            {
                "kind": "row",
                "index": 1,
                "source": "applicants.zip:table.xlsx",
                "text": "Name | Degree Field | Degree Level | Experience (Years) | Publications | Lab Trained (Y/N) | Citizen (Y/N) | Programming Lang | Second Language",
            },
            {
                "kind": "row",
                "index": 2,
                "source": "applicants.zip:table.xlsx",
                "text": "Hollie Wallace | Biotechnology | Master | 2 | 4 | Y | Y | C++ | Spanish",
            },
            {
                "kind": "row",
                "index": 3,
                "source": "applicants.zip:table.xlsx",
                "text": "Nabil Bates | Biology | Ph. D. | 4 | 1 | Y | Y | Fortran | Spanish",
            },
            {
                "kind": "row",
                "index": 4,
                "source": "applicants.zip:table.xlsx",
                "text": "Abi Haines | Biology | Master | 3 | 4 | Y | Y | C# | German",
            },
        ]

        answer, evidence = _solve_office_document_ops(
            "How many applicants for the job in the PDF are only missing a single qualification?",
            Path("applicants.zip"),
        )

        self.assertEqual(answer, "2")
        self.assertTrue(any("single-miss applicants=2" in item for item in evidence))

    @patch("domains.gaia_ops.backend._load_office_document_units")
    def test_office_document_ops_solves_secret_santa_missing_giver(self, mock_units: Any) -> None:
        mock_units.return_value = [
            {
                "kind": "paragraph",
                "index": 1,
                "source": "secret-santa.docx",
                "text": "\n".join(
                    [
                        "Employees",
                        "Harry",
                        "Rebecca",
                        "Georgette",
                        "Micah",
                        "Perry",
                        "Tyson",
                        "Lucy",
                        "Jun",
                        "Sara",
                        "Miguel",
                        "Fred",
                        "Alex",
                        "Gift Assignments",
                        "Giftee",
                        "Recipient",
                        "Harry",
                        "Miguel",
                        "Rebecca",
                        "Micah",
                        "Georgette",
                        "Lucy",
                        "Micah",
                        "Jun",
                        "Perry",
                        "Georgette",
                        "Tyson",
                        "Fred",
                        "Lucy",
                        "Alex",
                        "Jun",
                        "Harry",
                        "Sara",
                        "Perry",
                        "Fred",
                        "Rebecca",
                        "Miguel",
                        "Sara",
                        "Alex",
                        "Tyson",
                        "Profiles",
                        "Harry: Fishing, Camping, Wine",
                        "Rebecca: Cars, Dogs, Chocolate",
                        "Georgette: Yoga, Cooking, Green Energy",
                        "Micah: Knitting, Rainy Weather, Books",
                        "Perry: Old Movies, Rats, Journaling",
                        "Tyson: Historical Fiction Novels, Biking, Parakeets",
                        "Lucy: Coffee, Physics, Board Games",
                        "Jun: Woodworking, Barbecue, JavaScript",
                        "Sara: Tabletop RPGs, Spas, Music",
                        "Miguel: Astronomy, Decorative Washi Tape, Ketchup",
                        "Fred: Chemistry, Perl, Cats",
                        "Alex: Surfing, Audrey Hepburn, Manga",
                        "Gifts:",
                        "Galileo Galilei biography",
                        "Fishing reel",
                        "Raku programming guide",
                        "Chisel set",
                        "Custom dice",
                        "War and Peace American film copy",
                        "Yarn",
                        "One Piece graphic novel",
                        "War and Peace novel",
                        "Starbucks gift card",
                        "Foam exercise mat",
                    ]
                ),
            }
        ]

        answer, evidence = _solve_office_document_ops(
            "An office held a Secret Santa gift exchange. Based on the information in the document, who did not give a gift?",
            Path("secret-santa.docx"),
        )

        self.assertEqual(answer, "Fred")
        self.assertTrue(any("unmatched recipient=Rebecca" in item for item in evidence))

    @patch("domains.gaia_ops.backend._load_xlsx_workbook")
    def test_advanced_spreadsheet_ops_detects_missing_cycle(self, mock_workbook: Any) -> None:
        mock_workbook.return_value = {
            "sheets": [
                {
                    "name": "Sheet1",
                    "rows": [],
                    "cells": {
                        "A1": {"ref": "A1", "row": 1, "col": 1, "value": "", "formula": "", "fill": "00FF00"},
                        "A2": {"ref": "A2", "row": 2, "col": 1, "value": "", "formula": "", "fill": "00FF00"},
                        "B1": {"ref": "B1", "row": 1, "col": 2, "value": "", "formula": "", "fill": "00FF00"},
                    },
                }
            ],
            "sheet_map": {},
        }

        answer, evidence = _solve_advanced_spreadsheet_ops(
            "Green cells are plots owned by Earl Smith. Can Earl walk through every plot he owns and return to his starting plot without backtracking?",
            Path("plots.xlsx"),
        )

        self.assertEqual(answer, "No")
        self.assertTrue(any("cycle_exists=False" in item for item in evidence))

    @patch("domains.gaia_ops.backend._audio_transcript_segments")
    def test_audio_transcription_ops_extracts_phrase_from_timestamp_window(self, mock_segments: Any) -> None:
        mock_segments.return_value = [
            {"start": 28.0, "end": 32.0, "text": '"BLUE APPLE"'},
            {"start": 32.0, "end": 36.0, "text": "continues"},
        ]

        answer, evidence, provenance = _solve_audio_transcription_ops(
            "In the attached audio clip, what phrase is spoken at 30 seconds?",
            [Path("clip.mp3")],
        )

        self.assertEqual(answer, "BLUE APPLE")
        self.assertTrue(any("audio transcript answer=BLUE APPLE" in item for item in evidence))
        self.assertEqual(provenance, ["audio:clip.mp3", "audio:transcript"])

    @patch("domains.gaia_ops.backend._audio_transcript_segments")
    def test_audio_transcription_ops_counts_letter_occurrences(self, mock_segments: Any) -> None:
        mock_segments.return_value = [
            {"start": 0.0, "end": 4.0, "text": '"RED EEL"'},
        ]

        answer, evidence, provenance = _solve_audio_transcription_ops(
            'In the attached recording, how many times does the letter "E" appear in the spoken phrase?',
            [Path("letters.wav")],
        )

        self.assertEqual(answer, "3")
        self.assertTrue(any("audio transcript answer=3" in item for item in evidence))
        self.assertEqual(provenance, ["audio:letters.wav", "audio:transcript"])

    @patch("domains.gaia_ops.backend._audio_transcript_segments")
    def test_audio_transcription_ops_extracts_response_after_question(self, mock_segments: Any) -> None:
        mock_segments.return_value = [
            {"start": 10.0, "end": 12.0, "text": "Isn't that hot?"},
            {"start": 12.0, "end": 15.0, "text": "Indeed it is."},
        ]

        answer, evidence, provenance = _solve_audio_transcription_ops(
            'In the attached audio, what is said in response to the question "Isn\'t that hot?"',
            [Path("dialogue.m4a")],
        )

        self.assertEqual(answer, "Indeed it is")
        self.assertTrue(any("audio transcript answer=Indeed it is" in item for item in evidence))
        self.assertEqual(provenance, ["audio:dialogue.m4a", "audio:transcript"])

    @patch("domains.gaia_ops.backend._search_documents_for_title")
    def test_public_scalar_transform_ops_computes_difference(self, mock_search_title: Any) -> None:
        def _search_side_effect(title: str, *args: Any, **kwargs: Any) -> list[dict[str, str]]:
            if title == "Alpha City":
                return [{"title": title, "snippet": "", "url": "https://example.com/alpha", "text": "Alpha City population 120"}]
            return [{"title": title, "snippet": "", "url": "https://example.com/beta", "text": "Beta City population 90"}]

        mock_search_title.side_effect = _search_side_effect

        answer, evidence, provenance = _solve_public_scalar_transform_ops(
            'What is the difference between the populations of "Alpha City" and "Beta City" according to public reference sources?'
        )

        self.assertEqual(answer, "30")
        self.assertTrue(any("difference between Alpha City=120 and Beta City=90 => 30" in item for item in evidence))
        self.assertEqual(len(provenance), 2)

    @patch("domains.gaia_ops.backend._search_documents_for_title")
    def test_public_scalar_transform_ops_computes_percentage_ratio(self, mock_search_title: Any) -> None:
        def _search_side_effect(title: str, *args: Any, **kwargs: Any) -> list[dict[str, str]]:
            if title == "Alpha City":
                return [{"title": title, "snippet": "", "url": "https://example.com/alpha", "text": "Alpha City population 200"}]
            return [{"title": title, "snippet": "", "url": "https://example.com/beta", "text": "Beta City population 50"}]

        mock_search_title.side_effect = _search_side_effect

        answer, evidence, provenance = _solve_public_scalar_transform_ops(
            'What integer-rounded percentage of the population of "Alpha City" is the population of "Beta City"?'
        )

        self.assertEqual(answer, "25")
        self.assertTrue(any("percentage Beta City=50 / Alpha City=200 => 25" in item for item in evidence))
        self.assertEqual(len(provenance), 2)

    @patch("domains.gaia_ops.backend._search_documents_for_title")
    def test_public_scalar_transform_ops_computes_average(self, mock_search_title: Any) -> None:
        def _search_side_effect(title: str, *args: Any, **kwargs: Any) -> list[dict[str, str]]:
            mapping = {
                "Peak Alpha": 100.0,
                "Peak Beta": 200.0,
                "Peak Gamma": 300.0,
            }
            value = mapping[title]
            return [{"title": title, "snippet": "", "url": f"https://example.com/{title}", "text": f"{title} elevation {value}"}]

        mock_search_title.side_effect = _search_side_effect

        answer, evidence, provenance = _solve_public_scalar_transform_ops(
            'What is the average elevation of "Peak Alpha", "Peak Beta", and "Peak Gamma" according to public reference sources?'
        )

        self.assertEqual(answer, "200")
        self.assertTrue(any("average of 3 values => 200" in item for item in evidence))
        self.assertEqual(len(provenance), 3)

    @patch("domains.gaia_ops.backend._load_word_list_entries")
    def test_broad_symbolic_ops_solves_boggle_board(self, mock_word_list: Any) -> None:
        mock_word_list.return_value = (
            "brine",
            "brinies",
            "briniest",
            "briniest",
            "zebra",
        )

        answer, evidence, provenance = _solve_broad_symbolic_ops(
            """
            I thought we could try a fun word puzzle together :)

            I've got a Boggle board here:

            ABRL
            EITE
            IONS
            FPEI

            I'd like to know the longest word that can be generated from the board.
            Please find the longest English language word that can be generated from this board.
            If more than one word of the same length exists at the maximum word length, please report the longest word that comes first, alphabetically.
            Oh, and let's please just use the words_alpha dictionary found at https://github.com/dwyl/english-words as the dictionary for our game.
            """
        )

        self.assertEqual(answer, "briniest")
        self.assertTrue(any("boggle best word=briniest" in item for item in evidence))
        self.assertEqual(provenance, ["prompt:_solve_boggle_longest_word"])

    def test_broad_symbolic_ops_solves_adjacent_transposed_checksum(self) -> None:
        answer, evidence, provenance = _solve_broad_symbolic_ops(
            "The following numbers function similarly to ISBN 13 numbers, however, their validation methods are slightly different. Rather than using alternate weights of 1 and 3, the checksum digit is calculated with an alternate weight of 1 and some other positive integer less than 10. Otherwise, the checksum digit is calculated as expected. Unfortunately, there is an error in the data. Two adjacent columns have been transposed. These errored columns do not involve the final column or one of the first three columns. Using this information, please provide all potential solutions with the unknown weight and the smaller index of the two errored columns (assume we start our indexing at 0 and ignore hyphens). Give your answer in the form x, y where x is the weight and y is the smaller index of the two transposed columns. 978-354181391-9 978-946669746-1 978-398036139-6 978-447656680-4 978-279586664-7 978-595073693-3 978-976647652-6 978-591178125-5 978-728465924-5 978-414825155-9"
        )

        self.assertEqual(answer, "7, 9")
        self.assertTrue(any("checksum solutions=[(7, 9)]" in item for item in evidence))
        self.assertEqual(provenance, ["prompt:_solve_adjacent_transposed_checksum"])

    def test_broad_symbolic_ops_solves_logic_odd_one_out(self) -> None:
        answer, evidence, provenance = _solve_broad_symbolic_ops(
            "¬(A ∧ B) ↔ (¬A ∨ ¬B) ¬(A ∨ B) ↔ (¬A ∧ ¬B) (A → B) ↔ (¬B → ¬A) (A → B) ↔ (¬A ∨ B) (¬A → B) ↔ (A ∨ ¬B) ¬(A → B) ↔ (A ∧ ¬B) Which of the above is not logically equivalent to the rest? Provide the full statement that doesn't fit."
        )

        self.assertEqual(answer, "(¬A → B) ↔ (A ∨ ¬B)")
        self.assertTrue(any("logic mismatch counts=[1, 1, 1, 1, 5, 1]" in item for item in evidence))
        self.assertEqual(provenance, ["prompt:_solve_logic_odd_one_out"])

    def test_broad_symbolic_ops_solves_coin_box_minimax(self) -> None:
        answer, evidence, provenance = _solve_broad_symbolic_ops(
            "Bob was invited to participate in a game show, and he advanced to the final round. The final round offered Bob the chance to win a large sum by playing a game against the host. The host has 30 shiny prop coins, each of which is worth $1,000 if Bob manages to win them by playing the game. The host hides the coins in three different prize boxes and then shuffles their order. The only rule restricting the host's coin placement is that one box must contain at least 2 coins, and one box must contain 6 more coins than another box. In order to play, Bob must submit three guesses, one guess for the number of coins in each box. The box is then opened and the number of coins is revealed. If Bob's guess is a number greater than the number of coins in the box, Bob earns no coins. If Bob guesses a number equal to or less than the number of coins in the box, Bob wins a number of coins equal to his guess. If Bob plays uses the optimal strategy, what's the minimum amount of money he can win from the game?"
        )

        self.assertEqual(answer, "16000")
        self.assertTrue(any("best guarantee=16 coins with guesses=(8, 8, 8)" in item for item in evidence))
        self.assertEqual(provenance, ["prompt:_solve_coin_box_minimax"])

    def test_broad_symbolic_ops_solves_newton_prompt_with_math_delimiters(self) -> None:
        answer, evidence, provenance = _solve_broad_symbolic_ops(
            "Given $x_0 = -5$ and $f(x) = x^3 + 4x^2 - 3x + 8$, what is the smallest $n$ where using Newton's Method $n = n+1$ after rounding to four decimal places?"
        )

        self.assertEqual(answer, "2")
        self.assertTrue(any("x_2=" in item for item in evidence))
        self.assertEqual(provenance, ["prompt:_solve_newton_stability"])

    def test_temporal_anchor_uses_end_of_range_for_year_bounded_prompt(self) -> None:
        anchor = _temporal_anchor("How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)?")

        self.assertEqual(anchor.get("mode"), "year_range")
        self.assertEqual(anchor.get("start_year"), 2000)
        self.assertEqual(anchor.get("end_year"), 2009)
        self.assertEqual(anchor.get("year"), 2009)
        self.assertTrue(bool(anchor.get("historical")))

    def test_temporal_anchor_uses_pre_year_cutoff_for_historical_prompt(self) -> None:
        anchor = _temporal_anchor("What is the average number of pre-2020 works on the public researcher pages?")

        self.assertEqual(anchor.get("mode"), "before_year")
        self.assertEqual(anchor.get("boundary_year"), 2020)
        self.assertEqual(anchor.get("year"), 2019)
        self.assertEqual(anchor.get("end_year"), 2019)

    def test_temporal_anchor_prefers_explicit_snapshot_over_answer_range(self) -> None:
        anchor = _temporal_anchor(
            "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia."
        )

        self.assertEqual(anchor.get("mode"), "snapshot_year")
        self.assertEqual(anchor.get("year"), 2022)
        self.assertEqual(anchor.get("start_year"), 2022)
        self.assertEqual(anchor.get("end_year"), 2022)

    def test_temporal_anchor_maps_end_of_year_snapshot_prompt(self) -> None:
        anchor = _temporal_anchor(
            "On Ben & Jerry's online flavor graveyard as of end of 2022, what is the last line of the background rhyme under the flavor named after a common coding phrase?"
        )

        self.assertEqual(anchor.get("mode"), "snapshot_year")
        self.assertEqual(anchor.get("year"), 2022)
        self.assertEqual(anchor.get("month"), 12)
        self.assertEqual(anchor.get("day"), 31)

    def test_temporal_anchor_recognizes_end_of_day_snapshot_prompt(self) -> None:
        anchor = _temporal_anchor(
            "Use the pages as they appeared at the end of the day on July 3, 2023."
        )

        self.assertEqual(anchor.get("mode"), "snapshot_year")
        self.assertEqual(anchor.get("year"), 2023)
        self.assertEqual(anchor.get("month"), 7)
        self.assertEqual(anchor.get("day"), 3)
        self.assertTrue(bool(anchor.get("historical")))

    def test_temporal_query_variants_include_range_and_cutoff_hints(self) -> None:
        range_variants = _temporal_query_variants(
            "Mercedes Sosa discography",
            "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)?",
        )
        cutoff_variants = _temporal_query_variants(
            "ORCID works",
            "What is the average number of pre-2020 works on the public researcher pages?",
        )

        self.assertIn("Mercedes Sosa discography 2009", range_variants)
        self.assertIn("Mercedes Sosa discography 2000 2009", range_variants)
        self.assertIn("ORCID works before 2020", cutoff_variants)
        self.assertIn("ORCID works pre-2020", cutoff_variants)

    def test_extract_historical_navigation_title_ignores_leading_latest_phrase(self) -> None:
        title = _extract_historical_navigation_title(
            "What is the latest chronological year date written in the image on the webpage found when following the first citation reference link on the latest version of Carl Nebel's Wikipedia page as of August 2023?"
        )
        self.assertEqual(title, "Carl Nebel")

    def test_page_image_urls_prioritizes_gallery_content_over_toolbar_noise(self) -> None:
        urls = _page_image_urls(
            "http://web.archive.org/web/20170816145914/http://www.sloanrarebooks.com/Auctions/A22/item-nebel-voyage.html",
            """
            <html><body>
              <img src="https://web-static.archive.org/_static/images/loading.gif" />
              <a href="javascript:;" onclick="MM_openBrWindow('image.php?file=images/nebel-voyage-14.jpg','','width=850,height=650')">
                <img src="/web/20170816145914im_/http://www.sloanrarebooks.com/Auctions/A22/images/thumbnails/nebel-voyage-14_thumb.jpg" />
              </a>
              <img src="/web/20170816145914im_/http://www.sloanrarebooks.com/Auctions/A22/images/nebel-voyage-02.jpg" />
            </body></html>
            """,
        )

        self.assertTrue(urls)
        self.assertIn("image.php?file=images/nebel-voyage-14.jpg", urls[0])
        self.assertTrue(all("loading.gif" not in url for url in urls[:2]))

        def test_first_citation_reference_url_prefers_substantive_external_target(self) -> None:
                href = _first_citation_reference_url(
                        """
                        <html><body>
                            <ol class="references">
                                <li>
                                    <a href="https://web.archive.org/web/20230801000000/https://de.wikipedia.org/wiki/Carl_Nebel">archive mirror</a>
                                    <a href="https://www.sloanrarebooks.com/Auctions/A22/item-nebel-voyage.html">book dealer page</a>
                                </li>
                            </ol>
                        </body></html>
                        """
                )

                self.assertEqual(href, "https://www.sloanrarebooks.com/Auctions/A22/item-nebel-voyage.html")

    @patch("domains.gaia_ops.backend._solve_colored_number_statistics_image")
    def test_image_vision_ops_routes_statistics_prompt_to_color_solver(self, mock_stats: Any) -> None:
        mock_stats.return_value = ("17.056", ["red numbers=[1, 2]", "green numbers=[3, 4, 5]"])

        answer, evidence, provenance = _solve_image_vision_ops(
            "When you take the average of the standard population deviation of the red numbers and the standard sample deviation of the green numbers in this image using the statistics module in Python 3.11, what is the result rounded to the nearest three decimal points?",
            [Path("stats.png")],
        )

        self.assertEqual(answer, "17.056")
        self.assertTrue(any("red numbers=" in item for item in evidence))
        self.assertEqual(provenance, ["image:stats.png"])

    @patch("domains.gaia_ops.backend._solve_universal_ocr_reasoning")
    def test_image_vision_ops_delegates_to_universal_ocr_reasoning(self, mock_universal: Any) -> None:
        mock_universal.return_value = ("1/2,3/4", ["fractions from fractions.png: ['1/2', '3/4']"], ["image:fractions.png"])

        answer, evidence, provenance = _solve_image_vision_ops(
            "List the fractions shown in the image.",
            [Path("fractions.png")],
        )

        self.assertEqual(answer, "1/2,3/4")
        self.assertEqual(provenance, ["image:fractions.png"])
        mock_universal.assert_called_once_with(
            "List the fractions shown in the image.",
            local_paths=[Path("fractions.png")],
            allow_case_specific_heuristics=True,
        )

    @patch("domains.gaia_ops.backend._solve_universal_ocr_reasoning")
    def test_office_document_ops_delegates_to_universal_ocr_reasoning(self, mock_universal: Any) -> None:
        mock_universal.return_value = ("2025", ["years=[1998, 2007, 2023, 2025]"], ["office:report.pdf"])

        answer, evidence = _solve_office_document_ops(
            "What is the latest chronological year that appears in the attached report?",
            Path("report.pdf"),
        )

        self.assertEqual(answer, "2025")
        self.assertTrue(any("years=" in item for item in evidence))
        mock_universal.assert_called_once_with(
            "What is the latest chronological year that appears in the attached report?",
            local_paths=[Path("report.pdf")],
        )

    @patch("domains.gaia_ops.backend._ocr_image_url")
    @patch("domains.gaia_ops.backend._page_image_urls")
    @patch("domains.gaia_ops.backend._http_get_text")
    @patch("domains.gaia_ops.backend._historical_wikipedia_documents")
    @patch("domains.gaia_ops.backend._public_reference_title_candidates")
    def test_historical_reference_navigation_ops_uses_image_years_not_page_chrome(
        self,
        mock_titles: Any,
        mock_historical_docs: Any,
        mock_http_get: Any,
        mock_page_images: Any,
        mock_ocr_image: Any,
    ) -> None:
        mock_titles.return_value = ["Carl Nebel"]
        mock_historical_docs.return_value = [
            {
                "title": "Carl Nebel",
                "url": "https://en.wikipedia.org/wiki/Carl_Nebel",
                "html_text": '<html><body><ol class="references"><li><a href="https://example.com/reference-page">ref</a></li></ol></body></html>',
            }
        ]
        mock_http_get.return_value = "<html><body><footer>2026</footer><img src='historic.jpg' /></body></html>"
        mock_page_images.return_value = ["https://example.com/historic.jpg"]
        mock_ocr_image.return_value = ["Lithograph dated 1927"]

        answer, evidence, provenance = _solve_historical_reference_navigation_ops(
            "What is the latest chronological year date written in the image on the webpage found when following the first citation reference link on the latest version of Carl Nebel's Wikipedia page as of August 2023?"
        )

        self.assertEqual(answer, "1927")
        self.assertTrue(any("reference url=https://example.com/reference-page" in item for item in evidence))
        self.assertEqual(provenance, ["https://example.com/reference-page"])

    @patch("domains.gaia_ops.backend._solve_universal_ocr_reasoning")
    @patch("domains.gaia_ops.backend._historical_reference_navigation_sources")
    def test_historical_reference_navigation_ops_delegates_to_universal_ocr_reasoning(
        self,
        mock_sources: Any,
        mock_universal: Any,
    ) -> None:
        prompt = "What is the latest chronological year date written in the image on the webpage found when following the first citation reference link on Carl Nebel's Wikipedia page?"
        mock_sources.return_value = (["https://example.com/historic.jpg"], ["reference url=https://example.com/reference-page"], ["https://example.com/reference-page"])
        mock_universal.return_value = ("1927", ["years from historic.jpg: [1927]"], ["https://example.com/historic.jpg"])

        answer, evidence, provenance = _solve_historical_reference_navigation_ops(prompt)

        self.assertEqual(answer, "1927")
        self.assertEqual(provenance, ["https://example.com/reference-page"])
        self.assertTrue(any("reference url=https://example.com/reference-page" in item for item in evidence))
        mock_universal.assert_called_once_with(
            prompt,
            remote_image_urls=["https://example.com/historic.jpg"],
            allow_case_specific_heuristics=True,
        )

    @patch("domains.gaia_ops.backend._best_scalar_from_public_documents")
    @patch("domains.gaia_ops.backend._search_documents_for_title")
    def test_public_scalar_transform_ops_passes_anchor_prompt_to_title_search(
        self,
        mock_search_title: Any,
        mock_best_scalar: Any,
    ) -> None:
        prompt = 'What is the average elevation of "Peak Alpha" and "Peak Beta" as of 2019?'
        mock_search_title.return_value = [{"title": "stub", "snippet": "", "url": "https://example.com", "text": ""}]
        mock_best_scalar.return_value = ("100", [], "https://example.com")

        _solve_public_scalar_transform_ops(prompt)

        self.assertGreaterEqual(len(mock_search_title.call_args_list), 2)
        for call in mock_search_title.call_args_list:
            self.assertEqual(call.kwargs.get("anchor_prompt"), prompt)

    @patch("domains.gaia_ops.backend._solve_github_contributor_name_match", side_effect=AssertionError("legacy helper should not run"))
    @patch("domains.gaia_ops.backend._fetch_search_documents")
    def test_cross_source_entity_ops_matches_entity_across_github_and_public_reference(
        self, mock_search_docs: Any, _legacy_solver: Any
    ) -> None:
        def _fake_search(query: str, **_: Any) -> list[dict[str, str]]:
            if "same name as" in query.lower() or "github" in query.lower():
                return [
                    {
                        "url": "https://github.com/opencv/opencv/pull/1",
                        "title": "OpenCV PR by Zhao Ziyang",
                        "snippet": "Contributor Zhao Ziyang added Mask-RCNN support.",
                        "text": "Contributor Zhao Ziyang added Mask-RCNN support.",
                    }
                ]
            if "former chinese head of government" in query.lower():
                return [
                    {
                        "url": "https://en.wikipedia.org/wiki/Zhao_Ziyang",
                        "title": "Zhao Ziyang - former Chinese head of government",
                        "snippet": "Zhao Ziyang served as premier of China.",
                        "text": "Zhao Ziyang served as premier of China.",
                    }
                ]
            return []

        mock_search_docs.side_effect = _fake_search

        answer, evidence, provenance = _solve_cross_source_entity_ops(
            "Which contributor to the version of OpenCV where support was added for the Mask-RCNN model has the same name as a former Chinese head of government when the names are transliterated to the Latin alphabet?"
        )

        self.assertEqual(answer, "Zhao Ziyang")
        self.assertTrue(any("cross-source matched person" in item for item in evidence))
        self.assertEqual(
            provenance,
            ["https://github.com/opencv/opencv/pull/1", "https://en.wikipedia.org/wiki/Zhao_Ziyang"],
        )

    @patch("domains.gaia_ops.backend._solve_esther_prime_minister", side_effect=AssertionError("legacy helper should not run"), create=True)
    @patch("domains.gaia_ops.backend._search_documents_from_prompt")
    def test_cross_source_entity_ops_resolves_place_to_office_holder(self, mock_search_prompt: Any, _legacy_solver: Any) -> None:
        def _fake_search(query: str, **_: Any) -> list[dict[str, str]]:
            if "first named place" in query.lower():
                return [
                    {
                        "url": "https://example.com/esther-summary",
                        "title": "Book of Esther summary",
                        "snippet": "The first named place was India.",
                        "text": "In the Book of Esther, the first named place was India.",
                    }
                ]
            if "prime minister of india" in query.lower():
                return [
                    {
                        "url": "https://example.com/pm-india-1977",
                        "title": "Prime Minister of India in April 1977",
                        "snippet": "Morarji Desai was prime minister.",
                        "text": "Morarji Desai was prime minister of India in April 1977.",
                    }
                ]
            return []

        mock_search_prompt.side_effect = _fake_search

        answer, evidence, provenance = _solve_cross_source_entity_ops(
            "In the Book of Esther, what was the first named place? Who was the prime minister there in April 1977?"
        )

        self.assertEqual(answer, "Morarji Desai")
        self.assertTrue(any("place candidate=India" in item for item in evidence))
        self.assertEqual(provenance, ["https://example.com/esther-summary", "https://example.com/pm-india-1977"])

    @patch("domains.gaia_ops.backend._solve_british_museum_science_case", side_effect=AssertionError("legacy helper should not run"), create=True)
    @patch("domains.gaia_ops.backend._fetch_search_documents")
    def test_cross_source_entity_ops_joins_museum_object_to_paper_measurement(
        self, mock_search_docs: Any, _legacy_solver: Any
    ) -> None:
        def _fake_search(query: str, **_: Any) -> list[dict[str, str]]:
            if "museum" in query.lower():
                return [
                    {
                        "url": "https://britishmuseum.example/object/2012-5015-17",
                        "title": "British Museum object 2012,5015.17",
                        "snippet": "Shell species Tritia gibbosula",
                        "text": "Object 2012,5015.17 records the shell species Tritia gibbosula.",
                    }
                ]
            if "tritia gibbosula" in query.lower():
                return [
                    {
                        "url": "https://science.example/paper",
                        "title": "Science Advances shell beads paper",
                        "snippet": "Shell beads were 150 thousand years old.",
                        "text": "Science Advances reported shell beads were 150 thousand years old.",
                    }
                ]
            return []

        mock_search_docs.side_effect = _fake_search

        answer, evidence, provenance = _solve_cross_source_entity_ops(
            "According to the British Museum record with museum number 2012,5015.17, how many thousand years old was the shell species in the related Science Advances paper?"
        )

        self.assertEqual(answer, "150")
        self.assertTrue(any("species candidate=Tritia gibbosula" in item for item in evidence))
        self.assertEqual(provenance, ["https://britishmuseum.example/object/2012-5015-17", "https://science.example/paper"])

    def test_public_record_schedule_arrival_time_requires_clock_shaped_cell(self) -> None:
        documents = [
            {
                "url": "https://example.com/tri-rail",
                "html_text": """
                    <table>
                      <tr><th>Train</th><th>Passengers</th><th>Pompano Beach arrival</th></tr>
                      <tr><td>101</td><td>52618</td><td>6:41 PM</td></tr>
                      <tr><td>102</td><td>42000</td><td>51618</td></tr>
                    </table>
                """,
            }
        ]

        answer, evidence, provenance = _solve_public_record_schedule_arrival_time(
            "What time was the Tri-Rail train that carried the most passengers on May 27, 2019 scheduled to arrive in Pompano Beach?",
            documents,
        )

        self.assertEqual(answer, "6:41 PM")
        self.assertTrue(any("Pompano Beach arrival => 6:41 PM" in item for item in evidence))
        self.assertEqual(provenance, ["https://example.com/tri-rail"])

    def test_parse_service_daily_metric_line_splits_month_total_from_day_values(self) -> None:
        parsed = _parse_service_daily_metric_line(
            "P685 3,0850 0 0 352 360 0 0 0 0 0 355 372 0 0 0 0 0 337 349 0 0 0 0 0 319 311 330 0 0 0 0",
            31,
        )

        self.assertIsNotNone(parsed)
        service_id, total, days = parsed or ("", 0, [])
        self.assertEqual(service_id, "P685")
        self.assertEqual(total, 3085)
        self.assertEqual(days[0], 0)
        self.assertEqual(days[26], 330)
        self.assertEqual(len(days), 31)

    def test_select_best_solver_candidate_prefers_supported_historical_bundle(self) -> None:
        prompt = "What country had the least number of athletes at the 1928 Summer Olympics? Give the IOC country code as your answer."
        weak = _solver_candidate_bundle(
            "1896 1900 1904 1908 1912 1920 1924 1928",
            ["selected answer column Summer Olympic Games value=1896.0"],
            ["https://en.wikipedia.org/wiki/Summer_Olympic_Games"],
            method="broad_hub_page",
            source_bias=0.02,
        )
        strong = _solver_candidate_bundle(
            "CUB",
            ["parenthetical count candidate Cuba => 1", "mapped Cuba => CUB"],
            ["https://en.wikipedia.org/wiki/1928_Summer_Olympics"],
            method="exact_event_participation_list",
            source_bias=0.18,
        )

        answer, evidence, provenance = _select_best_solver_candidate(
            prompt,
            [weak, strong],
            research_mode="public_record_ops",
        )

        self.assertEqual(answer, "CUB")
        self.assertTrue(any("selected candidate via exact_event_participation_list" in item for item in evidence))
        self.assertEqual(provenance, ["https://en.wikipedia.org/wiki/1928_Summer_Olympics"])

    def test_select_best_solver_candidate_rejects_weak_bundle_set(self) -> None:
        prompt = "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)?"
        weak = _solver_candidate_bundle(
            "16",
            ["row count from section"],
            ["wikipedia:Mercedes Sosa"],
            method="weak_section_count",
            source_bias=0.0,
        )

        answer, evidence, provenance = _select_best_solver_candidate(
            prompt,
            [weak],
            research_mode="generic_public_reference",
            fallback_evidence=["generic public reference unresolved"],
        )

        self.assertEqual(answer, "")
        self.assertEqual(provenance, [])
        self.assertIn("generic public reference unresolved", evidence)

    def test_select_best_solver_candidate_prefers_person_name_for_person_prompt(self) -> None:
        prompt = "Assuming scientists in the famous youtube video were interviewed the same year, what is the name of the scientist predicting the sooner thinking machines or robots? Answer using the format First name Last name"
        title_bundle = _solver_candidate_bundle(
            "The Thinking Machine",
            ["video title=The Thinking Machine", "title mentioned near prediction language"],
            ["https://example.com/video"],
            method="video_title_match",
            source_bias=0.10,
            candidate_kind="short_text",
        )
        person_bundle = _solver_candidate_bundle(
            "Claude Shannon",
            ["video best person=Claude Shannon score=3.20", "prediction language near Claude Shannon"],
            ["https://example.com/video", "youtube:transcript"],
            method="video_person_evidence_graph",
            source_bias=0.08,
            candidate_kind="person_name",
        )

        answer, evidence, provenance = _select_best_solver_candidate(
            prompt,
            [title_bundle, person_bundle],
            research_mode="video_transcript_ops",
        )

        self.assertEqual(answer, "Claude Shannon")
        self.assertTrue(any("person-name fit" in item for item in evidence))
        self.assertEqual(provenance, ["https://example.com/video", "youtube:transcript"])

    def test_select_best_solver_candidate_contract_retry_skips_invalid_header_blob(self) -> None:
        prompt = "What horror movie is cited in the chapter? Answer with the exact movie title."
        invalid = _solver_candidate_bundle(
            "Name Rating Vacancy Pool Sample Review",
            ["table header blob from supporting page"],
            ["https://example.com/chapter"],
            method="header_blob_extract",
            source_bias=0.95,
            candidate_kind="short_text",
            answer_contract="title",
        )
        valid = _solver_candidate_bundle(
            "A Nightmare on Elm Street",
            ["quoted title near chapter citation", "movie title supported by chapter evidence"],
            ["https://example.com/chapter"],
            method="quoted_title_extract",
            source_bias=0.05,
            candidate_kind="short_text",
            answer_contract="title",
        )

        answer, evidence, provenance = _select_best_solver_candidate(
            prompt,
            [invalid, valid],
            research_mode="scholarly_reference_ops",
        )

        self.assertEqual(answer, "A Nightmare on Elm Street")
        self.assertTrue(any("selected candidate via quoted_title_extract" in item for item in evidence))
        self.assertEqual(provenance, ["https://example.com/chapter"])

    def test_plan_question_emits_gaia_runtime_metadata_and_snapshot(self) -> None:
        task_id = "runtime_plan_metadata_case"
        runtime_dir = Path(__file__).resolve().parents[1] / "logs" / "gaia_query_engine" / task_id
        if runtime_dir.exists():
            shutil.rmtree(runtime_dir)
        self.addCleanup(lambda: shutil.rmtree(runtime_dir, ignore_errors=True))

        state = SimpleNamespace(
            task_id=task_id,
            problem_text="What word repairs the expression? Answer with the missing token only.",
            metadata={
                "workspace_dir": str(Path(__file__).resolve().parents[1]),
                "workspace_files": [],
                "gaia_progress_logging": True,
                "gaia_resume_enabled": True,
            },
        )

        result = plan_question(state.problem_text, state)

        state_metadata = result["payload"]["state_metadata"]
        progress_path = Path(state_metadata["gaia_progress_log_path"])
        resume_path = Path(state_metadata["gaia_resume_snapshot_path"])

        self.assertEqual(state_metadata["gaia_runtime_stage"], "plan")
        self.assertIn("gaia_compact_state", state_metadata)
        self.assertTrue(state_metadata["gaia_recent_progress"])
        self.assertTrue(state_metadata["gaia_operator_registry"])
        self.assertTrue(progress_path.exists())
        self.assertTrue(resume_path.exists())

        snapshot = json.loads(resume_path.read_text(encoding="utf-8"))
        self.assertEqual(snapshot["task_id"], task_id)
        self.assertEqual(snapshot["stage"], "plan")
        self.assertIn("compact_state", snapshot)

    def test_best_person_name_from_documents_aggregates_across_all_documents(self) -> None:
        name, evidence = _best_person_name_from_documents(
            [
                {
                    "title": "Archive note",
                    "snippet": "",
                    "text": "The Thinking Machine featured Alan Turing.",
                },
                {
                    "title": "Companion article",
                    "snippet": "Claude Shannon made the prediction.",
                    "text": "Claude Shannon appears again in the transcript notes.",
                },
            ]
        )

        self.assertEqual(name, "Claude Shannon")
        self.assertTrue(any("Claude Shannon" in item for item in evidence))

    def test_plan_question_blind_mode_routes_public_species_lookup_structurally(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "I’m researching species that became invasive after people who kept them as pets released them. "
                "There’s a certain species of fish that was popularized as a pet by being the main character of the movie Finding Nemo. "
                "According to the USGS, where was this fish found as a nonnative species, before the year 2020?"
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "public_record_ops")
        self.assertIn("public records", result["result"])

    def test_plan_question_blind_mode_keeps_structural_family_routing(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "How many edits were made to the Wikipedia page on Antidisestablishmentarianism from its inception until June 2023?"
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "public_data_query_ops")
        self.assertEqual(question_plan.get("solver_submode"), "wikipedia_revision_count")

    def test_plan_question_routes_public_reference_history_ops_structurally(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "Who nominated the featured article candidacy for the latest 2022 English Wikipedia article about Lego?"
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "public_reference_history_ops")
        self.assertIn("revision/history sources", result["result"])

    def test_plan_question_routes_historical_web_prompt_to_public_reference_history_ops(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "On Ben & Jerry's online flavor graveyard as of end of 2022, what is the last line of the background rhyme under the flavor named after a common coding phrase?"
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "public_reference_history_ops")

    def test_plan_question_routes_public_record_ops_structurally(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer."
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]
        reasoning_schema = result["payload"]["state_metadata"]["reasoning_schema"]
        task_algebra = result["payload"]["state_metadata"]["task_algebra"]
        role_machine = result["payload"]["state_metadata"]["internal_role_machine"]

        self.assertEqual(question_plan.get("research_mode"), "public_record_ops")
        self.assertIn("public records", result["result"])
        self.assertEqual(reasoning_schema.get("source_family"), "public_record")
        self.assertEqual(reasoning_schema.get("output_contract"), "three_letter_code")
        self.assertIn("1928", reasoning_schema.get("time_anchor", ""))
        self.assertEqual(task_algebra.get("equation"), "time x source x operator x contract x rival")
        self.assertEqual(task_algebra.get("source_axis"), "public_record")
        self.assertEqual(role_machine.get("roles"), "framer -> retriever -> resolver -> judge -> closer")

    def test_plan_question_routes_public_data_density_submode_structurally(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "Use density measures from the chemistry materials licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials as compiled by LibreTexts. "
                "If I have a gallon of ethanol and a gallon of water, how many cups must I remove from the ethanol until it has less mass than the water?"
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "public_data_query_ops")
        self.assertEqual(question_plan.get("solver_submode"), "density_removal")

    def test_plan_question_routes_public_data_script_submode_structurally(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "In Series 9, Episode 11 of Doctor Who, the Doctor is trapped inside an ever-shifting maze. What is this location called in the official script for the episode? Give the setting exactly as it appears."
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "public_data_query_ops")
        self.assertEqual(question_plan.get("solver_submode"), "script_scene_heading")

    def test_plan_question_routes_public_data_revision_count_submode_structurally(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "How many edits were made to the Wikipedia page on Antidisestablishmentarianism from its inception until June 2023?"
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "public_data_query_ops")
        self.assertEqual(question_plan.get("solver_submode"), "wikipedia_revision_count")

    def test_plan_question_routes_public_data_wikipedia_link_distance_structurally(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "What is the minimum number of page links a person must click on to go from the English Wikipedia page on The Lord of the Rings to the English Wikipedia page on Frodo Baggins?"
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "public_data_query_ops")
        self.assertEqual(question_plan.get("solver_submode"), "wikipedia_link_distance")

    def test_plan_question_routes_public_data_usda_supersession_structurally(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "In July 2, 1959 United States standards for grades of processed fruits, vegetables, and certain other products listed as dehydrated, what integer-rounded percentage had newer USDA standards become effective by 2020?"
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "public_data_query_ops")
        self.assertEqual(question_plan.get("solver_submode"), "usda_standards_supersession")

    def test_plan_question_routes_paper_compare_ops_structurally(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                'What is the difference in measured time span between the papers "Paper A" and "Paper B"?'
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "scholarly_reference_ops")
        self.assertEqual(question_plan.get("solver_submode"), "paper_compare_ops")
        self.assertIn("cited scholarly source", result["result"])

    def test_plan_question_routes_video_transcript_ops_structurally(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "In the video https://www.youtube.com/watch?v=demo123, what command is clicked at 30 seconds?"
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "video_transcript_ops")
        self.assertIn("transcript evidence", result["result"])

    def test_plan_question_routes_youtube_video_without_explicit_url_to_video_transcript_ops(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "Assuming scientists in the famous youtube video The Thinking Machine (Artificial Intelligence in the 1960s) were interviewed the same year, "
                "what is the name of the scientist predicting the sooner thinking machines or robots? Answer using the format First name Last name"
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "video_transcript_ops")
        self.assertIn("transcript evidence", result["result"])

    def test_plan_question_routes_youtube_short_without_explicit_url_to_video_transcript_ops(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "What is the maximum length in meters of #9 in the first National Geographic short on YouTube that was ever released according to the Monterey Bay Aquarium website? Just give the number."
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "video_transcript_ops")
        self.assertIn("transcript evidence", result["result"])

    def test_plan_question_routes_youtube_page_website_lookup_to_generic_public_reference(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "Eva Draconis has a personal website which can be accessed on her YouTube page. "
                "What is the meaning of the only symbol seen in the top banner that has a curved line that isn't a circle or a portion of a circle? "
                "Answer without punctuation."
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "generic_public_reference")
        self.assertIn("referenced public page", result["result"])

    def test_plan_question_routes_web_archive_ops_structurally(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "Using the Wayback Machine, which menu item appeared on the restaurant website in March 2020 but no longer appeared in July 2020?"
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "web_archive_ops")
        self.assertIn("archived snapshots", result["result"])

    def test_plan_question_routes_historical_reference_navigation_ops_structurally(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "What is the latest chronological year date written in the image on the webpage found when following the first citation reference link on the latest version of Carl Nebel's Wikipedia page as of August 2023?"
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "historical_reference_navigation_ops")
        self.assertIn("historically anchored source page", result["result"])

    def test_plan_question_blind_mode_keeps_image_files_generic_until_solve(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "In the attached image, what is the latest chronological year that appears?"
                "\nWorkspace files:\n- poster.jpg"
            ),
            metadata={
                "workspace_files": ["poster.jpg"],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "poster.jpg",
                "candidate_files": ["poster.jpg"],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode", ""), "")
        self.assertIn("poster.jpg", result["result"])
        self.assertIn("solve intent=", result["result"])

    def test_plan_question_blind_mode_keeps_spreadsheets_generic_until_solve(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "Across all sheets in the attached workbook, which project has the highest score?"
                "\nWorkspace files:\n- scores.xlsx"
            ),
            metadata={
                "workspace_files": ["scores.xlsx"],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "scores.xlsx",
                "candidate_files": ["scores.xlsx"],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode", ""), "")
        self.assertEqual(question_plan.get("solver_submode", ""), "")
        self.assertIn("scores.xlsx", result["result"])
        self.assertIn("solve intent=", result["result"])

    @patch("domains.gaia_ops.backend._solve_spreadsheet_question")
    def test_solve_question_uses_generalized_spreadsheet_routing(self, mock_solver: Any) -> None:
        workbook_path = Path("tmp_spreadsheet_route.xlsx")
        workbook_path.write_text("placeholder", encoding="utf-8")
        mock_solver.return_value = ("Project Atlas", ["sheet-driven answer"])
        try:
            state = SimpleNamespace(
                problem_text=(
                    "Across all sheets in the attached workbook, which project has the highest score?"
                    "\nWorkspace files:\n- tmp_spreadsheet_route.xlsx"
                ),
                metadata={
                    "workspace_dir": str(Path.cwd()),
                    "workspace_files": ["tmp_spreadsheet_route.xlsx"],
                    "question_plan": {"research_mode": "spreadsheet_reasoning_ops", "solver_submode": "advanced_spreadsheet_ops"},
                    "candidate_files": ["tmp_spreadsheet_route.xlsx"],
                    "benchmark_assistance_mode": "unassisted",
                    "oracle_hints_enabled": False,
                },
            )

            result = solve_question(state.problem_text, state)
        finally:
            workbook_path.unlink(missing_ok=True)

        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"], "Project Atlas")
        mock_solver.assert_called_once()

    def test_plan_question_blind_mode_keeps_office_documents_generic_until_solve(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "In the attached presentation, what is the title on slide 2?"
                "\nWorkspace files:\n- deck.pptx"
            ),
            metadata={
                "workspace_files": ["deck.pptx"],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "deck.pptx",
                "candidate_files": ["deck.pptx"],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode", ""), "")
        self.assertIn("deck.pptx", result["result"])
        self.assertIn("solve intent=", result["result"])

    def test_plan_question_blind_mode_keeps_audio_files_generic_until_solve(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "In the attached audio clip, what phrase is spoken at 30 seconds?"
                "\nWorkspace files:\n- clip.mp3"
            ),
            metadata={
                "workspace_files": ["clip.mp3"],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "clip.mp3",
                "candidate_files": ["clip.mp3"],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode", ""), "")
        self.assertIn("clip.mp3", result["result"])
        self.assertIn("solve intent=", result["result"])

    @patch("domains.gaia_ops.backend._solve_office_document_ops")
    def test_solve_question_blind_mode_uses_generic_file_fallback_for_office_documents(self, mock_solver: Any) -> None:
        deck_path = Path("tmp_blind_office_route.pptx")
        deck_path.write_text("placeholder", encoding="utf-8")
        mock_solver.return_value = ("Slide Two", ["document fallback"])
        try:
            state = SimpleNamespace(
                problem_text=(
                    "In the attached presentation, what is the title on slide 2?"
                    "\nWorkspace files:\n- tmp_blind_office_route.pptx"
                ),
                metadata={
                    "workspace_dir": str(Path.cwd()),
                    "workspace_files": ["tmp_blind_office_route.pptx"],
                    "candidate_files": ["tmp_blind_office_route.pptx"],
                    "target_file": "tmp_blind_office_route.pptx",
                    "benchmark_assistance_mode": "unassisted",
                    "oracle_hints_enabled": False,
                    "allow_named_family_routing": False,
                    "blind_structural_mode": True,
                },
            )

            result = solve_question(state.problem_text, state)
        finally:
            deck_path.unlink(missing_ok=True)

        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"], "Slide Two")
        mock_solver.assert_called_once()

    def test_plan_question_routes_public_scalar_transform_ops_structurally(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                'What is the difference between the populations of "Alpha City" and "Beta City" according to public reference sources?'
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "public_data_query_ops")
        self.assertEqual(question_plan.get("solver_submode"), "public_scalar_transform_ops")
        self.assertIn("authoritative public sources", result["result"])

    def test_plan_question_routes_cross_source_entity_ops_structurally(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "Which contributor to the version of OpenCV where support was added for the Mask-RCNN model has the same name as a former Chinese head of government when the names are transliterated to the Latin alphabet?"
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "cross_source_entity_ops")
        self.assertIn("extract the key entity from the first source", result["result"])

    def test_plan_question_blind_mode_blocks_benchmark_shaped_family_routing(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "In the puzzle Pick that Ping-Pong, which ball should be selected to maximize the chance of winning?"
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode", ""), "text_reasoning_ops")
        self.assertEqual(question_plan.get("solver_submode"), "symbolic_reasoning_ops")
        self.assertIn("symbolic or combinatorial rules", result["result"])

    def test_plan_question_strict_generalized_blind_routes_public_record_from_coarse_source_cues(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "I’m researching species that became invasive after people who kept them as pets released them. "
                "There’s a certain species of fish that was popularized as a pet by being the main character of the movie Finding Nemo. "
                "According to the USGS, where was this fish found as a nonnative species, before the year 2020?"
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "allow_case_specific_heuristics": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "public_record_ops")
        self.assertTrue(question_plan.get("route_candidates"))
        self.assertGreaterEqual(float(question_plan.get("route_confidence", 0.0)), 0.48)
        self.assertEqual(question_plan.get("expected_evidence_kind"), "structured_public_record")

    def test_plan_question_strict_generalized_blind_keeps_structural_spreadsheet_file_routing(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "The attached spreadsheet contains a list of books I read in the year 2022. "
                "What is the title of the book that I read the slowest, using the rate of words per day?"
                "\nWorkspace files:\n- books.xlsx"
            ),
            metadata={
                "workspace_files": ["books.xlsx"],
                "allow_named_family_routing": False,
                "allow_case_specific_heuristics": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "spreadsheet_reasoning_ops")
        self.assertEqual(question_plan.get("solver_submode"), "spreadsheet_lookup")

    def test_plan_question_strict_generalized_blind_keeps_structural_document_file_routing(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "The attached file shows a list of books in the collection of Scribe County Public Library. "
                "How many of the library’s books that are authored by Rick Riordan are not currently on the library’s shelves?"
                "\nWorkspace files:\n- library.pdf"
            ),
            metadata={
                "workspace_files": ["library.pdf"],
                "allow_named_family_routing": False,
                "allow_case_specific_heuristics": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "office_document_ops")

    def test_plan_question_strict_generalized_blind_keeps_generic_scholarly_route(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                'What is the difference in measured time span between the papers "Paper A" and "Paper B"?'
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "allow_case_specific_heuristics": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "scholarly_reference_ops")
        self.assertEqual(question_plan.get("solver_submode"), "paper_compare_ops")
        self.assertEqual(question_plan.get("expected_evidence_kind"), "scholarly_source")

    def test_plan_question_strict_generalized_blind_routes_book_chapter_citation_prompt_to_scholarly(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                'In Valentina Re’s contribution to the 2017 book "World Building: Transmedia, Fans, Industries", '
                "what horror movie does the author cite as having popularized metalepsis between a dream world and reality?"
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "allow_case_specific_heuristics": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "scholarly_reference_ops")
        self.assertEqual(question_plan.get("solver_submode"), "quoted_paper_lookup")
        self.assertEqual(question_plan.get("expected_evidence_kind"), "scholarly_source")

    def test_plan_question_strict_generalized_blind_routes_github_prompt_from_source_family(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "According to GitHub, what date was the earliest closed issue with the label component: numpy.polynomial labeled as regression? Answer in MM/DD/YY."
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "allow_case_specific_heuristics": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "github_public_artifact_ops")
        self.assertTrue(question_plan.get("route_candidates"))
        self.assertEqual(question_plan.get("expected_evidence_kind"), "repository_record")

    def test_plan_question_strict_generalized_blind_routes_unlambda_prompt_generically(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                'In Unlambda, what exact character or text needs to be added to correct the following code to output "For penguins"? Code: r`````.F.o.r'
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "allow_case_specific_heuristics": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "text_reasoning_ops")
        self.assertEqual(question_plan.get("solver_submode"), "unlambda_missing_token")
        self.assertEqual(question_plan.get("expected_evidence_kind"), "prompt_text")

    def test_plan_question_strict_generalized_blind_routes_reversed_instruction_generically(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                '.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI'
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": False,
                "allow_case_specific_heuristics": False,
                "blind_structural_mode": True,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "text_reasoning_ops")
        self.assertEqual(question_plan.get("solver_submode"), "generic_text_reasoning")
        self.assertEqual(question_plan.get("expected_evidence_kind"), "prompt_text")

    def test_classify_no_file_source_families_does_not_treat_plain_release_verb_as_github_signal(self) -> None:
        candidates = _classify_no_file_source_families(
            "I’m researching species that became invasive after people who kept them as pets released them. "
            "According to the USGS, where was this fish found as a nonnative species, before the year 2020?"
        )

        self.assertTrue(candidates)
        self.assertEqual(candidates[0].get("research_mode"), "public_record_ops")
        self.assertEqual(candidates[0].get("expected_evidence_kind"), "structured_public_record")

    def test_plan_question_named_lane_still_canonicalizes_to_generalized_mode(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "In the puzzle Pick that Ping-Pong, which ball should be selected to maximize the chance of winning?"
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": True,
                "blind_structural_mode": False,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "text_reasoning_ops")
        self.assertEqual(question_plan.get("solver_submode"), "symbolic_reasoning_ops")
        self.assertIn("symbolic or combinatorial rules", result["result"])

    @patch("domains.gaia_ops.backend._solve_orcid_average_from_jsonld")
    def test_solve_question_blind_mode_disables_erratum_override(self, mock_orcid_solver: Any) -> None:
        workspace = Path(".tmp-tests") / "gaia-blind-erratum"
        workspace.mkdir(parents=True, exist_ok=True)
        jsonld_path = workspace / "bec74516-02fc-48dc-b202-55e78d0e17cf.jsonld"
        jsonld_path.write_text("{}", encoding="utf-8")
        mock_orcid_solver.return_value = ("17", ["mock average=17"])

        state = SimpleNamespace(
            task_id="bec74516-02fc-48dc-b202-55e78d0e17cf",
            problem_text="What is the average number of pre-2020 works on the open researcher and contributor identification pages of the people whose identification is in this file?",
            metadata={
                "workspace_dir": str(workspace),
                "workspace_files": [jsonld_path.name],
                "question_plan": {"research_mode": "orcid_jsonld_average"},
                "candidate_files": [jsonld_path.name],
                "benchmark_assistance_mode": "unassisted",
                "allow_errata_overrides": False,
            },
        )

        result = solve_question("", state)

        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"], "17")
        self.assertNotIn("benchmark:gaia-errata", result["payload"]["state_metadata"]["answer_provenance"])

    @patch("domains.gaia_ops.backend._solve_text_only_question")
    def test_solve_question_blind_mode_uses_generalized_symbolic_mode(
        self,
        mock_text_solver: Any,
    ) -> None:
        mock_text_solver.return_value = ("ball 3", ["symbolic route"], ["symbolic:choice"])
        state = SimpleNamespace(
            problem_text="In the puzzle Pick that Ping-Pong, which ball should be selected to maximize the chance of winning?",
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": [],
                "question_plan": {"research_mode": "text_reasoning_ops", "solver_submode": "symbolic_reasoning_ops"},
                "candidate_files": [],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
                "allow_named_family_routing": False,
                "blind_structural_mode": True,
            },
        )

        result = solve_question(state.problem_text, state)

        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"], "ball 3")
        mock_text_solver.assert_called_once()

    def test_plan_question_routes_legacy_esther_prompt_to_cross_source_entity_ops(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "In the Book of Esther, what was the first named place? Who was the prime minister there in April 1977?"
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": True,
                "blind_structural_mode": False,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "cross_source_entity_ops")

    def test_plan_question_routes_legacy_mercedes_prompt_to_generic_public_reference(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "How many Mercedes Sosa studio albums were published between 2000 and 2009 according to Wikipedia?"
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": True,
                "blind_structural_mode": False,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "generic_public_reference")

    def test_plan_question_routes_legacy_numpy_prompt_to_github_public_artifact_ops(self) -> None:
        state = SimpleNamespace(
            problem_text=(
                "According to GitHub, what date was the earliest closed issue with the label component: numpy.polynomial labeled as regression? Answer in MM/DD/YY."
                "\nWorkspace files:\n- none"
            ),
            metadata={
                "workspace_files": [],
                "allow_named_family_routing": True,
                "blind_structural_mode": False,
                "target_file": "",
                "candidate_files": [],
            },
        )

        result = plan_question("", state)
        question_plan = result["payload"]["state_metadata"]["question_plan"]

        self.assertEqual(question_plan.get("research_mode"), "github_public_artifact_ops")

    def test_classify_no_file_source_families_prefers_wikipedia_history_over_scholarly_article_wording(self) -> None:
        candidates = _classify_no_file_source_families(
            "How many images are there in the latest 2022 Lego english wikipedia article?"
        )

        self.assertTrue(candidates)
        self.assertEqual(candidates[0].get("research_mode"), "public_reference_history_ops")
        self.assertNotEqual(candidates[0].get("research_mode"), "scholarly_reference_ops")

    def test_classify_no_file_source_families_emits_answer_contract_and_operator_chain(self) -> None:
        candidates = _classify_no_file_source_families(
            "What country had the least number of athletes at the 1928 Summer Olympics? Give the IOC country code as your answer."
        )

        self.assertTrue(candidates)
        self.assertEqual(candidates[0].get("research_mode"), "public_record_ops")
        self.assertEqual(candidates[0].get("answer_contract"), "three_letter_code")
        self.assertIn("extract_structured_rows", list(candidates[0].get("operator_chain", [])))

    def test_build_plan_metadata_includes_operator_graph_and_contract(self) -> None:
        metadata = _build_plan_metadata(
            "What country had the least number of athletes at the 1928 Summer Olympics? Give the IOC country code as your answer.",
            "public_record_ops",
        )

        self.assertEqual(metadata["reasoning_schema"].get("answer_contract"), "three_letter_code")
        self.assertEqual(metadata["operator_graph"].get("primary_operator"), "discover_records")
        self.assertIn("normalize_answer", metadata["operator_graph"].get("operator_chain", ""))

    @patch("domains.gaia_ops.backend._public_record_search_documents")
    def test_generalized_browse_candidate_bundles_extract_table_code_answer(self, mock_search: Any) -> None:
        mock_search.return_value = [
            {
                "url": "https://example.com/olympics",
                "title": "1928 Summer Olympics medal table",
                "html_text": (
                    "<table><tr><th>Country</th><th>IOC</th><th>Athletes</th></tr>"
                    "<tr><td>Cuba</td><td>CUB</td><td>1</td></tr>"
                    "<tr><td>Peru</td><td>PER</td><td>2</td></tr></table>"
                ),
                "text": "Country IOC Athletes Cuba CUB 1 Peru PER 2",
            }
        ]

        bundles = _generalized_browse_candidate_bundles(
            "What country had the least number of athletes at the 1928 Summer Olympics? Give the IOC country code as your answer.",
            "public_record_ops",
        )

        self.assertTrue(any(bundle.get("candidate") == "CUB" for bundle in bundles))
        self.assertTrue(any(bundle.get("answer_contract") == "three_letter_code" for bundle in bundles))

    def test_validate_candidate_answer_rejects_url_for_title_prompt(self) -> None:
        accepted, _, report = _validate_candidate_answer(
            'Of the authors that worked on the paper "Pie Menus or Linear Menus, Which Is Better?" in 2015, what was the title of the first paper authored by the one that had authored prior papers?',
            "https://fau.digital.flvc.org/islandora/object/fau%3A4238",
            research_mode="scholarly_reference_ops",
            method="scholarly_reference_ops",
        )

        self.assertFalse(accepted)
        self.assertIn("url-shaped", " ".join(report.get("notes", [])))

    def test_validate_candidate_answer_rejects_non_numeric_for_numeric_prompt(self) -> None:
        accepted, _, report = _validate_candidate_answer(
            "I have the Standard plan in the image below. What is the average additional cost per file rounded to the nearest cent?",
            "Standard",
            research_mode="image_vision_ops",
            method="image_vision_ops",
        )

        self.assertFalse(accepted)
        self.assertIn("numeric", " ".join(report.get("notes", [])))

    def test_validate_candidate_answer_rejects_header_blob_for_short_text_prompt(self) -> None:
        accepted, _, report = _validate_candidate_answer(
            "The attached PDF lists accommodations in the resort community of Seahorse Island. Which type of accommodation has a higher average rating in Seahorse Island?",
            "Name Rating (out of 5) Vacancy Pool Sample Review",
            research_mode="office_document_ops",
            method="office_document_ops",
        )

        self.assertFalse(accepted)
        self.assertTrue(any("numeric" in note or "header" in note or "intermediate extraction" in note for note in report.get("notes", [])))

    def test_validate_candidate_answer_rejects_numeric_for_short_text_prompt(self) -> None:
        accepted, _, report = _validate_candidate_answer(
            "The attached PDF lists accommodations in the resort community of Seahorse Island. Which type of accommodation has a higher average rating in Seahorse Island?",
            "1",
            research_mode="office_document_ops",
            method="office_document_ops",
        )

        self.assertFalse(accepted)
        self.assertIn("short text", " ".join(report.get("notes", [])))

    def test_validate_candidate_answer_rejects_metadata_dump_for_title_prompt(self) -> None:
        accepted, _, report = _validate_candidate_answer(
            'Of the authors (First M. Last) that worked on the paper "Pie Menus or Linear Menus, Which Is Better?" in 2015, what was the title of the first paper authored by the one that had authored prior papers?',
            "paper authors=['Pie Menus', 'Or Linear', 'Which Is']",
            research_mode="scholarly_reference_ops",
            method="scholarly_reference_ops",
        )

        self.assertFalse(accepted)
        self.assertIn("title-like", " ".join(report.get("notes", [])))

    def test_validate_candidate_answer_rejects_integer_for_decimal_prompt(self) -> None:
        accepted, _, report = _validate_candidate_answer(
            "When you take the average of the standard population deviation of the red numbers and the standard sample deviation of the green numbers in this image using the statistics module in Python 3.11, what is the result rounded to the nearest three decimal points?",
            "4540",
            research_mode="image_vision_ops",
            method="image_vision_ops",
        )

        self.assertFalse(accepted)
        self.assertIn("decimal-form", " ".join(report.get("notes", [])))

    def test_validate_candidate_answer_rejects_institution_for_person_prompt(self) -> None:
        accepted, _, report = _validate_candidate_answer(
            "Assuming scientists in the famous youtube video were interviewed the same year, what is the name of the scientist? Answer using the format First name Last name",
            "Massachusetts Institute",
            research_mode="video_transcript_ops",
            method="video_transcript_ops",
        )

        self.assertFalse(accepted)
        self.assertIn("person name", " ".join(report.get("notes", [])))

    def test_validate_candidate_answer_rejects_non_identifier_for_contract_number_prompt(self) -> None:
        accepted, _, report = _validate_candidate_answer(
            "Under what NASA award number was the work performed by R. G. Arendt supported by?",
            "There Are Hundreds of Mysterious Filaments at the Center",
            research_mode="scholarly_reference_ops",
            method="scholarly_reference_ops",
        )

        self.assertFalse(accepted)
        self.assertIn("identifier-shaped", " ".join(report.get("notes", [])))

    def test_validate_candidate_answer_accepts_contract_number_identifier(self) -> None:
        accepted, normalized, report = _validate_candidate_answer(
            "Under what NASA award number was the work performed by R. G. Arendt supported by?",
            "80GSFC21M0002",
            research_mode="scholarly_reference_ops",
            method="scholarly_reference_ops",
        )

        self.assertTrue(accepted)
        self.assertEqual(normalized, "80GSFC21M0002")
        self.assertTrue(report.get("accepted"))

    def test_validate_candidate_answer_rejects_numeric_for_title_prompt(self) -> None:
        accepted, _, report = _validate_candidate_answer(
            'Of the authors (First M. Last) that worked on the paper "Pie Menus or Linear Menus, Which Is Better?" in 2015, what was the title of the first paper authored by the one that had authored prior papers?',
            "325509",
            research_mode="scholarly_reference_ops",
            method="scholarly_reference_ops",
        )

        self.assertFalse(accepted)
        self.assertIn("title-like", " ".join(report.get("notes", [])))

    def test_validate_candidate_answer_rejects_title_echo_of_quoted_source(self) -> None:
        accepted, _, report = _validate_candidate_answer(
            'In Valentina Re’s contribution to the 2017 book "World Building: Transmedia, Fans, Industries", '
            "what horror movie does the author cite as having popularized metalepsis between a dream world and reality?",
            "Industries Mapping",
            research_mode="scholarly_reference_ops",
            method="scholarly_reference_ops",
        )

        self.assertFalse(accepted)
        self.assertIn("title-like", " ".join(report.get("notes", [])))

    @patch("domains.gaia_ops.backend._solve_public_record_ops")
    def test_solve_question_quality_control_rejects_non_time_answer_for_time_prompt(self, mock_solver: Any) -> None:
        mock_solver.return_value = ("52618", ["max Passengers=52618"], ["https://example.com/tri-rail"])
        state = SimpleNamespace(
            problem_text="What time was the Tri-Rail train that carried the most passengers on May 27, 2019 scheduled to arrive in Pompano Beach?",
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": [],
                "question_plan": {"research_mode": "public_record_ops"},
                "candidate_files": [],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
            },
        )

        result = solve_question(state.problem_text, state)

        self.assertFalse(result["ok"])

    @patch("domains.gaia_ops.backend._solve_video_transcript_ops")
    def test_solve_question_quality_control_rejects_title_when_prompt_requires_person_name(self, mock_solver: Any) -> None:
        mock_solver.return_value = ("The Thinking Machine", ["video title=The Thinking Machine"], ["https://example.com/video"])
        state = SimpleNamespace(
            problem_text="What is the name of the scientist predicting the future of AI? Answer using the format First name Last name",
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": [],
                "question_plan": {"research_mode": "video_transcript_ops"},
                "candidate_files": [],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
            },
        )

        result = solve_question(state.problem_text, state)

        self.assertFalse(result["ok"])
        self.assertIn("quality checks", result["result"])
        self.assertIn("quality checks", result["result"])

    @patch("domains.gaia_ops.backend._solve_video_transcript_ops")
    def test_solve_question_quality_control_rejects_boilerplate_navigation_text_for_person_prompt(self, mock_solver: Any) -> None:
        mock_solver.return_value = (
            "About Press Copyright",
            ["video best person=About Press Copyright"],
            ["https://example.com/video"],
        )
        state = SimpleNamespace(
            problem_text="What is the name of the scientist predicting the future of AI? Answer using the format First name Last name",
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": [],
                "question_plan": {"research_mode": "video_transcript_ops"},
                "candidate_files": [],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
            },
        )

        result = solve_question(state.problem_text, state)

        self.assertFalse(result["ok"])
        self.assertIn("quality checks", result["result"])

    @patch("domains.gaia_ops.backend._solve_image_vision_ops")
    def test_solve_question_quality_control_rejects_raw_ocr_fragment_for_numeric_image_prompt(self, mock_solver: Any) -> None:
        mock_solver.return_value = ("archive_prefix", ["ocr from image"], ["image:script.png"])
        state = SimpleNamespace(
            problem_text="The attached image contains a Python script. Run the Python code and return the sum of the third and fifth integers in the sorted list.",
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": ["script.png"],
                "question_plan": {"research_mode": "image_vision_ops"},
                "candidate_files": ["script.png"],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
                "allow_case_specific_heuristics": False,
            },
        )

        result = solve_question(state.problem_text, state)

        self.assertFalse(result["ok"])
        self.assertIn("quality checks", result["result"])

    @patch("domains.gaia_ops.backend._solve_office_document_ops")
    def test_solve_question_quality_control_rejects_office_header_blob(self, mock_solver: Any) -> None:
        workspace = Path(".tmp-tests") / "gaia-office-quality"
        workspace.mkdir(parents=True, exist_ok=True)
        pdf_path = workspace / "stay.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")
        mock_solver.return_value = ("Name Rating (out of 5) Vacancy Pool Sample Review", ["document header blob"])
        state = SimpleNamespace(
            problem_text="The attached PDF lists accommodations in the resort community of Seahorse Island. Which type of accommodation has a higher average rating in Seahorse Island?",
            metadata={
                "workspace_dir": str(workspace),
                "workspace_files": [pdf_path.name],
                "question_plan": {"research_mode": "office_document_ops"},
                "candidate_files": [pdf_path.name],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
                "allow_case_specific_heuristics": False,
            },
        )

        result = solve_question(state.problem_text, state)

        self.assertFalse(result["ok"])
        self.assertIn("quality checks", result["result"])

    def test_solve_question_quality_control_rejects_raw_first_line_file_fallback(self) -> None:
        workspace = Path(".tmp-tests") / "gaia-fallback-quality"
        workspace.mkdir(parents=True, exist_ok=True)
        script_path = workspace / "program.py"
        script_path.write_text("from random import randint\nprint(0)\n", encoding="utf-8")
        state = SimpleNamespace(
            problem_text="What is the final numeric output from the attached Python code?",
            metadata={
                "workspace_dir": str(workspace),
                "workspace_files": [script_path.name],
                "candidate_files": [script_path.name],
                "target_file": script_path.name,
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
                "allow_case_specific_heuristics": False,
            },
        )

        result = solve_question(state.problem_text, state)

        self.assertFalse(result["ok"])
        self.assertIn("quality checks", result["result"])

    @patch("domains.gaia_ops.backend._solve_public_record_ops")
    def test_solve_question_strict_generalized_blind_uses_public_record_family_with_generic_flag(self, mock_solver: Any) -> None:
        mock_solver.return_value = ("CUB", ["metric column=Athletes"], ["https://example.com/olympics"])
        state = SimpleNamespace(
            problem_text="What country had the least number of athletes at the 1928 Summer Olympics? Give the IOC country code as your answer.",
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": [],
                "question_plan": {"research_mode": "public_record_ops"},
                "candidate_files": [],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
                "allow_case_specific_heuristics": False,
            },
        )

        result = solve_question(state.problem_text, state)

        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"], "CUB")
        mock_solver.assert_called_once_with(
            state.problem_text,
            allow_case_specific_heuristics=False,
        )

    @patch("domains.gaia_ops.backend._solve_public_scalar_transform_ops", side_effect=AssertionError("generic scalar path should not run"))
    @patch("domains.gaia_ops.backend._solve_wikipedia_revision_count")
    def test_solve_question_strict_generalized_blind_uses_public_data_submode_operator(
        self,
        mock_revision_solver: Any,
        _scalar_solver: Any,
    ) -> None:
        mock_revision_solver.return_value = ("123", ["revision count=123"])
        state = SimpleNamespace(
            problem_text="How many edits were made to the Wikipedia page on Antidisestablishmentarianism from its inception until June 2023?",
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": [],
                "question_plan": {
                    "research_mode": "public_data_query_ops",
                    "solver_submode": "wikipedia_revision_count",
                },
                "candidate_files": [],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
                "allow_case_specific_heuristics": False,
            },
        )

        result = solve_question(state.problem_text, state)

        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"], "123")
        mock_revision_solver.assert_called_once_with(state.problem_text)

    @patch("domains.gaia_ops.backend._solve_broad_symbolic_ops")
    def test_solve_question_strict_generalized_blind_skips_broad_symbolic_helper(self, mock_solver: Any) -> None:
        state = SimpleNamespace(
            problem_text="In the puzzle Pick that Ping-Pong, which ball should be selected to maximize the chance of winning?",
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": [],
                "question_plan": {},
                "candidate_files": [],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
                "allow_case_specific_heuristics": False,
            },
        )

        result = solve_question(state.problem_text, state)

        self.assertFalse(result["ok"])
        mock_solver.assert_not_called()

    @patch("domains.gaia_ops.backend._solve_generic_public_reference")
    @patch("domains.gaia_ops.backend._solve_cross_source_entity_ops")
    @patch("domains.gaia_ops.backend._solve_video_transcript_ops")
    def test_solve_question_external_ensemble_recovers_from_empty_primary_candidate(
        self,
        mock_video_solver: Any,
        mock_cross_source_solver: Any,
        mock_generic_solver: Any,
    ) -> None:
        mock_video_solver.return_value = ("", ["transcript mentions prediction but no resolved speaker"], ["https://example.com/video"])
        mock_cross_source_solver.return_value = (
            "Claude Shannon",
            ["entity bridge -> Claude Shannon", "prediction language near Claude Shannon"],
            ["https://example.com/video", "https://example.com/companion"],
        )
        mock_generic_solver.return_value = ("", [], [])
        state = SimpleNamespace(
            problem_text="What is the name of the scientist predicting the future of AI? Answer using the format First name Last name",
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": [],
                "question_plan": {"research_mode": "video_transcript_ops"},
                "candidate_files": [],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
            },
        )

        result = solve_question(state.problem_text, state)

        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"], "Claude Shannon")
        self.assertEqual(result["payload"]["state_metadata"]["candidate_answer"], "Claude Shannon")
        self.assertIn("https://example.com/companion", result["payload"]["state_metadata"]["answer_provenance"])

    @patch("domains.gaia_ops.backend._solve_public_record_ops")
    def test_solve_question_quality_control_accepts_valid_time_answer(self, mock_solver: Any) -> None:
        mock_solver.return_value = ("6:41 pm", ["Pompano Beach arrival => 6:41 pm"], ["https://example.com/2019/tri-rail"])
        state = SimpleNamespace(
            problem_text="What time was the Tri-Rail train that carried the most passengers on May 27, 2019 scheduled to arrive in Pompano Beach? Express your answer in the 12-hour digital clock format with AM or PM.",
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": [],
                "question_plan": {"research_mode": "public_record_ops"},
                "candidate_files": [],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
            },
        )

        result = solve_question(state.problem_text, state)

        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"], "6:41 PM")

    @patch("domains.gaia_ops.backend._solve_scholarly_reference_ops")
    def test_solve_question_quality_control_rejects_numeric_answer_for_textual_prompt(self, mock_solver: Any) -> None:
        mock_solver.return_value = ("10.4324", ["targeted numeric match -> 10.4324"], ["https://example.com/chapter"])
        state = SimpleNamespace(
            problem_text='In Valentina Re’s contribution to the 2017 book "World Building: Transmedia, Fans, Industries", what horror movie does the author cite as having popularized metalepsis between a dream world and waking life?',
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": [],
                "question_plan": {"research_mode": "scholarly_reference_ops", "solver_submode": "quoted_paper_lookup"},
                "candidate_files": [],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
            },
        )

        result = solve_question(state.problem_text, state)

        self.assertFalse(result["ok"])
        self.assertIn("quality checks", result["result"])

    @patch("domains.gaia_ops.backend._fallback_external_solver_bundles", return_value=[])
    @patch("domains.gaia_ops.backend._resolve_scholarly_documents")
    @patch("domains.gaia_ops.backend._solve_scholarly_reference_ops")
    def test_solve_question_uses_generalized_browse_bundles_when_primary_scholarly_solver_is_empty(
        self,
        mock_solver: Any,
        mock_documents: Any,
        _mock_fallback: Any,
    ) -> None:
        mock_solver.return_value = ("", ["scholarly source unresolved"], [])
        mock_documents.return_value = [
            {
                "url": "https://example.com/chapter",
                "title": "World Building chapter",
                "text": (
                    "Re discusses dream/reality metalepsis and states that A Nightmare on Elm Street popularized "
                    "the device between a dream world and waking life."
                ),
                "html_text": "",
                "pdf_text": "",
                "combined_text": (
                    "Re discusses dream/reality metalepsis and states that A Nightmare on Elm Street popularized "
                    "the device between a dream world and waking life."
                ),
            }
        ]
        state = SimpleNamespace(
            problem_text='In Valentina Re’s contribution to the 2017 book "World Building: Transmedia, Fans, Industries", what horror movie does the author cite as having popularized metalepsis between a dream world and waking life?',
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": [],
                "question_plan": {"research_mode": "scholarly_reference_ops", "solver_submode": "quoted_paper_lookup"},
                "candidate_files": [],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
                "allow_case_specific_heuristics": False,
            },
        )

        result = solve_question(state.problem_text, state)

        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"], "A Nightmare on Elm Street")
        self.assertEqual(
            result["payload"]["state_metadata"]["reasoning_schema"].get("answer_contract"),
            "title",
        )

    @patch("domains.gaia_ops.backend._solve_public_record_ops")
    def test_solve_question_quality_control_rejects_non_code_for_ioc_prompt(self, mock_solver: Any) -> None:
        mock_solver.return_value = (
            "1896 1900 1904 1908 1912 1920 1924 1928",
            ["selected answer column Year value=1896 1900 1904 1908 1912 1920 1924 1928"],
            ["https://example.com/olympics"],
        )
        state = SimpleNamespace(
            problem_text="What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer.",
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": [],
                "question_plan": {"research_mode": "public_record_ops"},
                "candidate_files": [],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
            },
        )

        result = solve_question(state.problem_text, state)

        self.assertFalse(result["ok"])
        self.assertIn("quality checks", result["result"])

    @patch("domains.gaia_ops.backend._wikipedia_backlinks")
    @patch("domains.gaia_ops.backend._wikipedia_page_links")
    def test_wikipedia_link_distance_solver_uses_graph_distance(self, mock_links: Any, mock_backlinks: Any) -> None:
        graph = {
            "The Lord of the Rings": ["Fantasy", "Middle-earth", "J. R. R. Tolkien"],
            "Fantasy": ["A Song of Ice and Fire"],
            "Middle-earth": [],
            "J. R. R. Tolkien": [],
            "A Song of Ice and Fire": [],
        }
        backlinks = {
            "A Song of Ice and Fire": ["Fantasy"],
            "Fantasy": ["The Lord of the Rings"],
        }
        mock_links.side_effect = lambda title: graph.get(title, [])
        mock_backlinks.side_effect = lambda title: backlinks.get(title, [])

        answer, evidence = _solve_wikipedia_link_distance(
            "What is the minimum number of page links a person must click on to go from the english Wikipedia page on "
            "The Lord of the Rings (the book) to the english Wikipedia page on A Song of Ice and Fire (the book series)? "
            "In your count, include each link you would click."
        )

        self.assertEqual(answer, "2")
        self.assertTrue(any("path depth=2" in item for item in evidence))

    @patch("domains.gaia_ops.backend._wikipedia_backlinks_as_of")
    @patch("domains.gaia_ops.backend._wikipedia_page_links_as_of")
    def test_wikipedia_link_distance_solver_uses_historical_bidirectional_search(
        self,
        mock_links_as_of: Any,
        mock_backlinks_as_of: Any,
    ) -> None:
        graph = {
            "The Lord of the Rings": ("Fantasy", "Middle-earth"),
            "Fantasy": ("A Song of Ice and Fire",),
            "Middle-earth": tuple(),
            "A Song of Ice and Fire": tuple(),
        }
        backlinks = {
            "A Song of Ice and Fire": ("Fantasy",),
            "Fantasy": ("The Lord of the Rings",),
        }
        mock_links_as_of.side_effect = lambda title, timestamp: graph.get(title, tuple())
        mock_backlinks_as_of.side_effect = lambda title, timestamp: backlinks.get(title, tuple())

        answer, evidence = _solve_wikipedia_link_distance(
            "What is the minimum number of page links a person must click on to go from the english Wikipedia page on "
            "The Lord of the Rings (the book) to the english Wikipedia page on A Song of Ice and Fire (the book series)? "
            "In your count, include each link you would click on to get to the page. "
            "Use the pages as they appeared at the end of the day on July 3, 2023."
        )

        self.assertEqual(answer, "2")
        self.assertTrue(any("historical cutoff=20230703" in item for item in evidence))
        mock_links_as_of.assert_any_call("The Lord of the Rings", "20230703")
        mock_backlinks_as_of.assert_any_call("A Song of Ice and Fire", "20230703")

    @patch("domains.gaia_ops.backend._wikipedia_query")
    @patch("domains.gaia_ops.backend._wikipedia_revision_id_as_of")
    def test_wikipedia_page_links_as_of_caches_and_prunes_titles(
        self,
        mock_revision_id: Any,
        mock_query: Any,
    ) -> None:
        _wikipedia_page_links_as_of.cache_clear()
        mock_revision_id.return_value = (12345, "2023-07-03T23:59:59Z")
        mock_query.return_value = {
            "parse": {
                "links": [
                    {"ns": 0, "*": "Fantasy"},
                    {"ns": 4, "*": "Wikipedia:About"},
                    {"ns": 0, "*": "A Song of Ice and Fire#Adaptations"},
                ]
            }
        }

        try:
            first = _wikipedia_page_links_as_of("The Lord of the Rings", "20230703")
            second = _wikipedia_page_links_as_of("The Lord of the Rings", "20230703")
        finally:
            _wikipedia_page_links_as_of.cache_clear()

        self.assertEqual(first, ("Fantasy", "A Song of Ice and Fire"))
        self.assertEqual(second, first)
        mock_revision_id.assert_called_once_with("The Lord of the Rings", "20230703")
        mock_query.assert_called_once()

    @patch("domains.gaia_ops.backend._wikipedia_backlinks")
    @patch("domains.gaia_ops.backend._wikipedia_page_links")
    def test_wikipedia_bidirectional_link_distance_reports_budget_exhaustion(
        self,
        mock_links: Any,
        mock_backlinks: Any,
    ) -> None:
        mock_links.side_effect = lambda title: {
            "Alpha": ["Beta", "Gamma"],
            "Beta": ["Delta"],
            "Gamma": ["Epsilon"],
        }.get(title, [])
        mock_backlinks.side_effect = lambda title: {
            "Omega": ["Psi", "Chi"],
            "Psi": ["Upsilon"],
            "Chi": ["Tau"],
        }.get(title, [])

        answer, evidence = _wikipedia_bidirectional_link_distance(
            "Alpha",
            "Omega",
            expansion_budget=1,
            time_budget_seconds=30.0,
        )

        self.assertEqual(answer, "")
        self.assertTrue(any("search budget exhausted" in item for item in evidence))

    @patch("domains.gaia_ops.backend._wikipedia_revision_count_until")
    def test_wikipedia_revision_count_solver_uses_cutoff(self, mock_revision_count: Any) -> None:
        mock_revision_count.return_value = 2732

        answer, evidence = _solve_wikipedia_revision_count(
            "How many edits were made to the Wikipedia page on Antidisestablishmentarianism from its inception until June 2023?"
        )

        self.assertEqual(answer, "2732")
        self.assertTrue(any("2732" in item for item in evidence))

    @patch("domains.gaia_ops.backend._wikipedia_revision_count_until")
    def test_wikipedia_revision_count_solver_accepts_month_of_year_form(self, mock_revision_count: Any) -> None:
        mock_revision_count.return_value = 2732

        answer, evidence = _solve_wikipedia_revision_count(
            "How many edits were made to the Wikipedia page on Antidisestablishmentarianism from its inception until June of 2023?"
        )

        self.assertEqual(answer, "2732")
        self.assertTrue(any("revision cutoff=2023-06-30T23:59:59Z" in item for item in evidence))

    @patch("domains.gaia_ops.backend._geocode_zip")
    @patch("domains.gaia_ops.backend._public_record_search_documents")
    def test_solve_public_record_ops_extracts_zip_from_structured_agency_record(
        self,
        mock_search_documents: Any,
        mock_geocode_zip: Any,
    ) -> None:
        mock_search_documents.return_value = [
            {
                "title": "clown anemonefish (Amphiprion ocellaris) - Collection record",
                "url": "https://nas.er.usgs.gov/queries/SpecimenViewer.aspx?SpecimenID=1468741",
                "text": (
                    "Specimen Information Amphiprion ocellaris (clown anemonefish) "
                    "State FL County Pinellas Locality Gulf of America, Florida, Fred Howard Park "
                    "Collection Year 2018"
                ),
                "html_text": "",
                "snippet": "",
            }
        ]
        mock_geocode_zip.return_value = "34689"

        answer, evidence, provenance = _solve_public_record_ops(
            "I’m researching species that became invasive after people who kept them as pets released them. "
            "There’s a certain species of fish that was popularized as a pet by being the main character of the movie Finding Nemo. "
            "According to the USGS, where was this fish found as a nonnative species, before the year 2020? "
            "I need the answer formatted as the five-digit zip codes of the places the species was found.",
            allow_case_specific_heuristics=False,
        )

        self.assertEqual(answer, "34689")
        self.assertTrue(any("Fred Howard Park" in item for item in evidence))
        self.assertEqual(provenance, ["https://nas.er.usgs.gov/queries/SpecimenViewer.aspx?SpecimenID=1468741"])

    @patch("domains.gaia_ops.backend._public_reference_title_candidates")
    @patch("domains.gaia_ops.backend._historical_wikipedia_documents")
    def test_generic_public_reference_counts_year_bounded_entries_from_section_html(self, mock_historical_docs: Any, mock_titles: Any) -> None:
        mock_titles.return_value = ["Mercedes Sosa discography"]
        mock_historical_docs.return_value = [
            {
                "title": "Mercedes Sosa discography",
                "url": "https://en.wikipedia.org/w/index.php?oldid=20220901000000",
                "html_text": """
                <html><body>
                <h2>Studio albums</h2>
                <ul>
                  <li>1999 - Earlier Work</li>
                  <li>2000 - Acústico</li>
                  <li>2003 - Corazón Libre</li>
                  <li>2011 - Deja la vida volar</li>
                </ul>
                </body></html>
                """,
                "text": "discography",
                "wikitext": "",
            }
        ]

        answer, evidence, provenance = _solve_generic_public_reference(
            "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia."
        )

        self.assertEqual(answer, "2")
        self.assertTrue(any("Mercedes Sosa discography" in item for item in evidence))
        self.assertEqual(provenance, ["https://en.wikipedia.org/w/index.php?oldid=20220901000000"])

    @patch("domains.gaia_ops.backend._public_reference_title_candidates")
    @patch("domains.gaia_ops.backend._historical_wikipedia_documents")
    def test_generic_public_reference_selects_argmin_from_public_table(self, mock_historical_docs: Any, mock_titles: Any) -> None:
        mock_titles.return_value = ["1928 Summer Olympics"]
        mock_historical_docs.return_value = [
            {
                "title": "1928 Summer Olympics",
                "url": "https://en.wikipedia.org/w/index.php?oldid=19280000000000",
                "html_text": """
                <html><body>
                <table class="wikitable">
                  <tr><th>Nation</th><th>IOC code</th><th>Athletes</th></tr>
                  <tr><td>Cuba</td><td>CUB</td><td>1</td></tr>
                  <tr><td>Dominican Republic</td><td>DOM</td><td>1</td></tr>
                  <tr><td>United States</td><td>USA</td><td>100</td></tr>
                </table>
                </body></html>
                """,
                "text": "olympics",
                "wikitext": "",
            }
        ]

        answer, evidence, provenance = _solve_generic_public_reference(
            "What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer."
        )

        self.assertEqual(answer, "CUB")
        self.assertTrue(any("metric column" in item for item in evidence))
        self.assertEqual(provenance, ["wikipedia:1928 Summer Olympics"])

    @patch("domains.gaia_ops.backend._public_reference_title_candidates")
    @patch("domains.gaia_ops.backend._historical_wikipedia_documents")
    def test_generic_public_reference_returns_adjacent_roster_names(self, mock_historical_docs: Any, mock_titles: Any) -> None:
        mock_titles.return_value = ["Hokkaido Nippon-Ham Fighters"]
        mock_historical_docs.return_value = [
            {
                "title": "Hokkaido Nippon-Ham Fighters",
                "url": "https://en.wikipedia.org/w/index.php?oldid=20230701000000",
                "html_text": """
                <html><body>
                <table class="wikitable">
                  <tr><th>Number</th><th>Pitcher</th></tr>
                  <tr><td>18</td><td>Kazunari Yoshida</td></tr>
                  <tr><td>19</td><td>Taishō Tamai</td></tr>
                  <tr><td>20</td><td>Kenta Uehara</td></tr>
                </table>
                </body></html>
                """,
                "text": "roster",
                "wikitext": "",
            }
        ]

        answer, evidence, provenance = _solve_generic_public_reference(
            "Who are the pitchers with the number before and after Taishō Tamai's number as of July 2023? Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
        )

        self.assertEqual(answer, "Yoshida, Uehara")
        self.assertTrue(any("number=19" in item for item in evidence))
        self.assertEqual(provenance, ["wikipedia:Hokkaido Nippon-Ham Fighters"])

    @patch("domains.gaia_ops.backend._public_reference_title_candidates")
    @patch("domains.gaia_ops.backend._historical_wikipedia_documents")
    def test_generic_public_reference_counts_images_on_public_page(self, mock_historical_docs: Any, mock_titles: Any) -> None:
        mock_titles.return_value = ["Lego"]
        mock_historical_docs.return_value = [
            {
                "title": "Lego",
                "url": "https://en.wikipedia.org/w/index.php?oldid=20220101000000",
                "html_text": """
                <html><body>
                <div class="mw-parser-output">
                  <img src="https://example.com/a.jpg" alt="brick" />
                  <img src="https://example.com/b.jpg" alt="set" />
                  <img src="https://example.com/c.jpg" alt="logo" />
                </div>
                </body></html>
                """,
                "text": "lego",
                "wikitext": "",
            }
        ]

        answer, evidence, provenance = _solve_generic_public_reference(
            "How many images are there in the latest 2022 Lego english wikipedia article?"
        )

        self.assertEqual(answer, "3")
        self.assertTrue(any("image count" in item for item in evidence))
        self.assertEqual(provenance, ["https://en.wikipedia.org/w/index.php?oldid=20220101000000"])

    @patch("domains.gaia_ops.backend._public_reference_search_documents")
    @patch("domains.gaia_ops.backend._public_reference_title_candidates")
    def test_generic_public_reference_uses_search_document_tables_when_title_candidates_are_empty(
        self, mock_titles: Any, mock_search_docs: Any
    ) -> None:
        mock_titles.return_value = []
        mock_search_docs.return_value = [
            {
                "title": "Olympics mirror",
                "url": "https://example.com/olympics",
                "text": "mirror page",
                "html_text": """
                <html><body>
                <table>
                  <tr><th>Nation</th><th>IOC code</th><th>Athletes</th></tr>
                  <tr><td>Cuba</td><td>CUB</td><td>1</td></tr>
                  <tr><td>United States</td><td>USA</td><td>100</td></tr>
                </table>
                </body></html>
                """,
            }
        ]

        answer, evidence, provenance = _solve_generic_public_reference(
            "What country had the least number of athletes at the 1928 Summer Olympics? Give the IOC country code as your answer."
        )

        self.assertEqual(answer, "CUB")
        self.assertTrue(any("url=https://example.com/olympics" in item for item in evidence))
        self.assertEqual(provenance, ["https://example.com/olympics"])

    @patch("domains.gaia_ops.backend._http_get_text")
    @patch("domains.gaia_ops.backend._fetch_search_documents")
    def test_public_record_search_documents_prioritizes_exact_event_title(
        self,
        mock_fetch_search: Any,
        mock_http_get_text: Any,
    ) -> None:
        mock_fetch_search.return_value = [
            {
                "title": "Summer Olympic Games - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Summer_Olympic_Games",
                "text": "broad olympics page",
            },
            {
                "title": "1928 Summer Olympics - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/1928_Summer_Olympics",
                "text": "exact event page",
            },
        ]
        mock_http_get_text.return_value = "<html><body>placeholder</body></html>"

        documents = _public_record_search_documents(
            "What country had the least number of athletes at the 1928 Summer Olympics? Give the IOC country code as your answer."
        )

        self.assertGreaterEqual(len(documents), 2)
        self.assertEqual(documents[0]["title"], "1928 Summer Olympics - Wikipedia")

    @patch("domains.gaia_ops.backend._fetch_search_documents")
    def test_public_record_search_queries_expand_with_resolved_species_candidates(self, mock_fetch_search: Any) -> None:
        def _fake_search(query: str, **_: Any) -> list[dict[str, str]]:
            if "Finding Nemo" in query:
                return [
                    {
                        "title": "Clown anemonefish",
                        "url": "https://example.com/species",
                        "snippet": "The clown anemonefish (Amphiprion ocellaris) is the species popularized by the film.",
                        "text": "The clown anemonefish (Amphiprion ocellaris) is the species popularized by the film.",
                    }
                ]
            return []

        mock_fetch_search.side_effect = _fake_search

        queries = _public_record_search_queries(
            "I’m researching species that became invasive after people who kept them as pets released them. "
            "There’s a certain species of fish that was popularized as a pet by being the main character of the movie Finding Nemo. "
            "According to the USGS, where was this fish found as a nonnative species, before the year 2020?"
        )

        self.assertIn("USGS clown anemonefish nonnative species", queries)
        self.assertIn("USGS Amphiprion ocellaris collection record", queries)

    @patch("domains.gaia_ops.backend._fetch_search_documents")
    def test_cross_source_entity_ops_restricts_same_name_primary_lookup_to_github_domain(self, mock_search_docs: Any) -> None:
        calls: list[tuple[str, dict[str, Any]]] = []

        def _fake_search(query: str, **kwargs: Any) -> list[dict[str, str]]:
            calls.append((query, dict(kwargs)))
            if "github" in query.lower():
                return [
                    {
                        "url": "https://github.com/opencv/opencv/issues/11412",
                        "title": "Support Mask RCNN models · Issue #11412 · opencv/opencv - GitHub",
                        "snippet": "Issue discussion mentions Li Peng among contributors.",
                        "text": "Li Peng appears in the issue discussion as a contributor name candidate.",
                    }
                ]
            if "former chinese head of government" in query.lower():
                return [
                    {
                        "url": "https://example.com/china-leaders",
                        "title": "Leaders of China",
                        "snippet": "Li Peng served as premier.",
                        "text": "Li Peng served as premier of China.",
                    }
                ]
            return []

        mock_search_docs.side_effect = _fake_search

        answer, evidence, provenance = _solve_cross_source_entity_ops(
            "Which contributor to the version of OpenCV where support was added for the Mask-RCNN model has the same name as a former Chinese head of government when the names are transliterated to the Latin alphabet?"
        )

        self.assertEqual(answer, "Li Peng")
        self.assertTrue(any(kwargs.get("allow_domains") == ("github.com",) for query, kwargs in calls if "github" in query.lower()))
        self.assertTrue(any("cross-source matched person=Li Peng" in item for item in evidence))
        self.assertEqual(provenance, ["https://github.com/opencv/opencv/issues/11412", "https://example.com/china-leaders"])

    @patch("domains.gaia_ops.backend._wayback_snapshot_url")
    @patch("domains.gaia_ops.backend._http_get_text")
    @patch("domains.gaia_ops.backend._fetch_search_documents")
    def test_search_documents_for_title_prefers_historical_year_aligned_result(
        self,
        mock_fetch_search: Any,
        mock_http_get_text: Any,
        mock_wayback_url: Any,
    ) -> None:
        mock_fetch_search.return_value = [
            {
                "title": "Mercedes Sosa discography",
                "url": "https://en.wikipedia.org/wiki/Mercedes_Sosa_discography",
                "snippet": "Current discography page",
                "text": "Current discography page",
            },
            {
                "title": "Mercedes Sosa discography (2009 archive)",
                "url": "https://example.com/mercedes-sosa-2009",
                "snippet": "Historical 2009 discography snapshot",
                "text": "Historical 2009 discography snapshot",
            },
        ]
        mock_http_get_text.return_value = "<html><body>discography</body></html>"
        mock_wayback_url.return_value = ""

        documents = _search_documents_for_title(
            "Mercedes Sosa discography",
            suffix_terms=("wikipedia",),
            anchor_prompt="How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)?",
        )

        self.assertGreaterEqual(len(documents), 2)
        self.assertEqual(documents[0]["url"], "https://example.com/mercedes-sosa-2009")

    @patch("domains.gaia_ops.backend._wayback_snapshot_url")
    @patch("domains.gaia_ops.backend._http_get_text")
    @patch("domains.gaia_ops.backend._fetch_search_documents")
    def test_public_reference_search_documents_adds_archived_snapshot_for_historical_prompt(
        self,
        mock_fetch_search: Any,
        mock_http_get_text: Any,
        mock_wayback_url: Any,
    ) -> None:
        prompt = "As of July 2023, who are the pitchers before and after Taisho Tamai?"
        current_url = "https://en.wikipedia.org/wiki/Example_roster"
        archive_url = "https://web.archive.org/web/20230715000000/https://en.wikipedia.org/wiki/Example_roster"
        mock_fetch_search.return_value = [
            {
                "title": "Example roster - Wikipedia",
                "url": current_url,
                "snippet": "Current roster",
                "text": "Current roster",
            }
        ]

        def _http_side_effect(url: str, *args: Any, **kwargs: Any) -> str:
            if url == current_url:
                return "<html><body>Current roster with enough surrounding text to exceed the archive materialization threshold for historical anchoring.</body></html>"
            if url == archive_url:
                return "<html><body>Archived 2023 roster with enough surrounding text to exceed the archive materialization threshold for historical anchoring.</body></html>"
            return ""

        mock_http_get_text.side_effect = _http_side_effect
        mock_wayback_url.return_value = archive_url

        documents = _search_documents_from_prompt(prompt, suffix_terms=("wikipedia",))

        self.assertTrue(any(document.get("url") == archive_url for document in documents))

    @patch("domains.gaia_ops.backend._public_reference_search_documents")
    @patch("domains.gaia_ops.backend._historical_wikipedia_documents")
    @patch("domains.gaia_ops.backend._public_reference_title_candidates")
    def test_public_reference_history_ops_uses_historical_snapshot_for_year_count(
        self,
        mock_titles: Any,
        mock_historical_docs: Any,
        mock_search_docs: Any,
    ) -> None:
        mock_titles.return_value = ["Mercedes Sosa discography"]
        mock_historical_docs.return_value = [
            {
                "title": "Mercedes Sosa discography",
                "url": "https://en.wikipedia.org/w/index.php?oldid=20220901000000",
                "html_text": """
                    <html><body>
                    <h2>Studio albums</h2>
                    <table>
                      <tr><th>Year</th><th>Album</th></tr>
                      <tr><td>2000</td><td>Misa criolla</td></tr>
                      <tr><td>2003</td><td>Acústico</td></tr>
                      <tr><td>2009</td><td>Cantora</td></tr>
                      <tr><td>2011</td><td>Posthumous</td></tr>
                    </table>
                    </body></html>
                """,
                "text": "discography",
                "wikitext": "",
            }
        ]
        mock_search_docs.return_value = []

        answer, evidence, provenance = _solve_public_reference_history_ops(
            "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia."
        )

        self.assertEqual(answer, "3")
        self.assertTrue(any("structured table count 2000-2009: 3" in item for item in evidence))
        self.assertEqual(provenance, ["https://en.wikipedia.org/w/index.php?oldid=20220901000000"])

    @patch("domains.gaia_ops.backend._public_reference_search_documents")
    @patch("domains.gaia_ops.backend._historical_wikipedia_documents")
    @patch("domains.gaia_ops.backend._public_reference_title_candidates")
    def test_public_reference_history_ops_does_not_abort_when_search_fallback_fails(
        self,
        mock_titles: Any,
        mock_historical_docs: Any,
        mock_search_docs: Any,
    ) -> None:
        mock_titles.return_value = ["Mercedes Sosa discography"]
        mock_historical_docs.return_value = [
            {
                "title": "Mercedes Sosa discography",
                "url": "https://en.wikipedia.org/w/index.php?oldid=20220901000000",
                "html_text": """
                    <html><body>
                    <h2>Studio albums</h2>
                    <table>
                      <tr><th>Year</th><th>Album</th></tr>
                      <tr><td>2000</td><td>Misa criolla</td></tr>
                      <tr><td>2003</td><td>Acústico</td></tr>
                      <tr><td>2009</td><td>Cantora</td></tr>
                    </table>
                    </body></html>
                """,
                "text": "discography",
                "wikitext": "",
            }
        ]
        mock_search_docs.side_effect = RuntimeError("search unavailable")

        answer, evidence, provenance = _solve_public_reference_history_ops(
            "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia."
        )

        self.assertEqual(answer, "3")
        self.assertTrue(any("structured table count 2000-2009: 3" in item for item in evidence))
        self.assertEqual(provenance, ["https://en.wikipedia.org/w/index.php?oldid=20220901000000"])

    @patch("domains.gaia_ops.backend._public_reference_search_documents")
    @patch("domains.gaia_ops.backend._historical_wikipedia_documents")
    @patch("domains.gaia_ops.backend._wikipedia_revision_snapshots_around")
    @patch("domains.gaia_ops.backend._public_reference_title_candidates")
    def test_public_reference_history_ops_extracts_removed_phrase_from_revision_snapshots(
        self,
        mock_titles: Any,
        mock_snapshots: Any,
        mock_historical_docs: Any,
        mock_search_docs: Any,
    ) -> None:
        mock_titles.return_value = ["Example Dragon"]
        mock_historical_docs.return_value = []
        mock_snapshots.return_value = [
            {"timestamp": "2023-06-10T10:00:00Z", "content": "* Silly dragon joke\n* Stable line"},
            {"timestamp": "2023-06-12T10:00:00Z", "content": "* Stable line"},
        ]
        mock_search_docs.return_value = []

        answer, evidence, provenance = _solve_public_reference_history_ops(
            "What phrase was removed from the Wikipedia page on Example Dragon on June 11, 2023?"
        )

        self.assertEqual(answer, "Silly dragon joke")
        self.assertTrue(any("removed phrase" in item for item in evidence))
        self.assertEqual(provenance, ["wikipedia:Example Dragon", "wikipedia:revisions"])

    @patch("domains.gaia_ops.backend._public_reference_search_documents")
    @patch("domains.gaia_ops.backend._public_reference_title_candidates")
    def test_public_reference_history_ops_extracts_featured_article_nominator(
        self,
        mock_titles: Any,
        mock_search_docs: Any,
    ) -> None:
        mock_titles.return_value = []
        mock_search_docs.return_value = [
            {
                "title": "Featured article candidate",
                "url": "https://en.wikipedia.org/wiki/Wikipedia:Featured_article_candidates/Lego/archive1",
                "text": "Nominated by Jane Smith on 12 May 2022.",
                "html_text": "<html><body><p>Nominated by Jane Smith on 12 May 2022.</p></body></html>",
            }
        ]

        answer, evidence, provenance = _solve_public_reference_history_ops(
            "Who nominated the featured article candidacy for the latest 2022 English Wikipedia article about Lego?"
        )

        self.assertEqual(answer, "Jane Smith")
        self.assertTrue(any("nominator" in item for item in evidence))
        self.assertEqual(provenance, ["https://en.wikipedia.org/wiki/Wikipedia:Featured_article_candidates/Lego/archive1"])

    @patch("domains.gaia_ops.backend._public_record_search_documents")
    def test_public_record_ops_counts_stops_between_named_stations(self, mock_search_docs: Any) -> None:
        mock_search_docs.return_value = [
            {
                "title": "Franklin/Foxboro Line",
                "url": "https://example.com/franklin-line",
                "text": "South Station | Back Bay | Ruggles | Forest Hills | Windsor Gardens",
                "html_text": """
                <html><body>
                <table>
                  <tr><th>Station</th></tr>
                  <tr><td>South Station</td></tr>
                  <tr><td>Back Bay</td></tr>
                  <tr><td>Ruggles</td></tr>
                  <tr><td>Forest Hills</td></tr>
                  <tr><td>Windsor Gardens</td></tr>
                </table>
                </body></html>
                """,
            }
        ]

        answer, evidence, provenance = _solve_public_record_ops(
            "How many stations are between South Station and Windsor Gardens on the Franklin/Foxboro Line?"
        )

        self.assertEqual(answer, "3")
        self.assertTrue(any("ordered" in item for item in evidence))
        self.assertEqual(provenance, ["https://example.com/franklin-line"])

    @patch("domains.gaia_ops.backend._public_record_search_documents")
    def test_public_record_ops_extracts_first_name_for_defunct_nationality(self, mock_search_docs: Any) -> None:
        mock_search_docs.return_value = [
            {
                "title": "Competition winners",
                "url": "https://example.com/malko",
                "text": "",
                "html_text": """
                <html><body>
                <table>
                  <tr><th>Year</th><th>Recipient</th><th>Nationality</th></tr>
                  <tr><td>1980</td><td>Alexei Petrov</td><td>Soviet Union</td></tr>
                  <tr><td>1988</td><td>Maria Jensen</td><td>Denmark</td></tr>
                </table>
                </body></html>
                """,
            }
        ]

        answer, evidence, provenance = _solve_public_record_ops(
            "What was the first name of the only recipient after 1977 with a defunct nationality in the 20th century?"
        )

        self.assertEqual(answer, "Alexei")
        self.assertTrue(any("defunct nationality row" in item for item in evidence))
        self.assertEqual(provenance, ["https://example.com/malko"])

    @patch("domains.gaia_ops.backend._public_record_schedule_documents")
    @patch("domains.gaia_ops.backend._public_record_search_documents")
    def test_public_record_ops_extracts_arrival_time_from_schedule_table(
        self,
        mock_search_docs: Any,
        mock_schedule_docs: Any,
    ) -> None:
        mock_search_docs.return_value = [
            {
                "title": "Tri-Rail ridership",
                "url": "https://example.com/tri-rail",
                "text": "",
                "html_text": """
                <html><body>
                <table>
                  <tr><th>Train</th><th>Passengers</th><th>Pompano Beach Arrival</th></tr>
                  <tr><td>TR1</td><td>84</td><td>8:14 AM</td></tr>
                  <tr><td>TR2</td><td>121</td><td>9:02 AM</td></tr>
                </table>
                </body></html>
                """,
            }
        ]
        mock_schedule_docs.side_effect = lambda prompt, seed_documents, service_id="": list(seed_documents)

        answer, evidence, provenance = _solve_public_record_ops(
            "What time was the Tri-Rail train that carried the most passengers on May 27, 2019 scheduled to arrive in Pompano Beach?"
        )

        self.assertEqual(answer, "9:02 AM")
        self.assertTrue(any("Passengers" in item for item in evidence))
        self.assertEqual(provenance, ["https://example.com/tri-rail"])

    @patch("domains.gaia_ops.backend._public_record_schedule_documents")
    @patch("domains.gaia_ops.backend._public_record_search_documents")
    def test_public_record_ops_joins_daily_report_to_schedule_tables(
        self,
        mock_search_docs: Any,
        mock_schedule_docs: Any,
    ) -> None:
        documents = [
            {
                "title": "Tri-Rail Commuter Rail Operations",
                "url": "https://example.com/ridership.pdf",
                "text": "",
                "html_text": "",
                "pdf_text": (
                    "Report for: May 2019\n"
                    "P685 3,0850 0 0 352 360 0 0 0 0 0 355 372 0 0 0 0 0 337 349 0 0 0 0 0 319 311 330 0 0 0 0\n"
                    "P680 2,9670 0 0 358 349 0 0 0 0 0 334 324 0 0 0 0 0 379 273 0 0 0 0 0 332 302 316 0 0 0 0\n"
                ),
            },
            {
                "title": "Weekend/Holiday Train Schedule",
                "url": "https://example.com/schedule",
                "text": "",
                "pdf_text": "",
                "html_text": """
                    <html><body>
                    <table>
                      <tr><th>Weekend-Southbound</th></tr>
                      <tr><td>Mangonia Park</td></tr>
                      <tr><td>West Palm Beach</td></tr>
                      <tr><td>Lake Worth Beach</td></tr>
                      <tr><td>Boynton Beach</td></tr>
                      <tr><td>Delray Beach</td></tr>
                      <tr><td>Boca Raton</td></tr>
                      <tr><td>Deerfield Beach</td></tr>
                      <tr><td>Pompano Beach</td></tr>
                    </table>
                    <table>
                      <tr><th>P685</th><th>P680</th></tr>
                      <tr><td>5:20 PM</td><td>2:52 PM</td></tr>
                      <tr><td>5:26 PM</td><td>2:57 PM</td></tr>
                      <tr><td>5:36 PM</td><td>3:03 PM</td></tr>
                      <tr><td>5:43 PM</td><td>3:09 PM</td></tr>
                      <tr><td>5:52 PM</td><td>3:14 PM</td></tr>
                      <tr><td>5:59 PM</td><td>3:23 PM</td></tr>
                      <tr><td>6:06 PM</td><td>3:27 PM</td></tr>
                      <tr><td>6:41 PM</td><td>3:31 PM</td></tr>
                    </table>
                    </body></html>
                """,
            },
        ]
        mock_search_docs.return_value = documents
        mock_schedule_docs.side_effect = lambda prompt, seed_documents, service_id="": list(seed_documents)

        answer, evidence, provenance = _solve_public_record_ops(
            "What time was the Tri-Rail train that carried the most passengers on May 27, 2019 scheduled to arrive in Pompano Beach? Express your answer in the 12-hour digital clock format with AM or PM."
        )

        self.assertEqual(answer, "6:41 PM")
        self.assertTrue(any("selected service=P685" in item for item in evidence))
        self.assertTrue(any("paired schedule tables" in item for item in evidence))
        self.assertEqual(provenance, ["https://example.com/ridership.pdf", "https://example.com/schedule"])

    @patch("domains.gaia_ops.backend._wikipedia_rendered_text")
    @patch("domains.gaia_ops.backend._wikipedia_search_titles")
    @patch("domains.gaia_ops.backend._public_record_search_documents")
    def test_public_record_ops_prefers_exact_participation_list_over_broad_olympics_page(
        self,
        mock_search_docs: Any,
        mock_search_titles: Any,
        mock_rendered_text: Any,
    ) -> None:
        mock_search_docs.return_value = [
            {
                "title": "Summer Olympic Games - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Summer_Olympic_Games",
                "text": "1896 1900 1904 1908 1912 1920 1924 1928 1932",
                "html_text": """
                <html><body>
                <table>
                  <tr><th>Summer Olympic Games</th></tr>
                  <tr><td>1896</td></tr>
                  <tr><td>1900</td></tr>
                  <tr><td>1904</td></tr>
                  <tr><td>1908</td></tr>
                </table>
                </body></html>
                """,
            },
            {
                "title": "1928 Summer Olympics - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/1928_Summer_Olympics",
                "text": "Argentina (81 athletes) Cuba (1) Dominican Republic (1) United States (280)",
                "html_text": """
                <html><body>
                <table>
                  <tr><th>Participating National Olympic Committees</th></tr>
                  <tr><td>Argentina (81 athletes) Cuba (1) Dominican Republic (1) United States (280)</td></tr>
                </table>
                </body></html>
                """,
            },
        ]
        mock_search_titles.return_value = ["Cuba at the 1928 Summer Olympics"]
        mock_rendered_text.return_value = "Nation Cuba NOC CUB Games 1928"

        answer, evidence, provenance = _solve_public_record_ops(
            "What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer."
        )

        self.assertEqual(answer, "CUB")
        self.assertTrue(any("parenthetical count candidate Cuba => 1" in item for item in evidence))
        self.assertTrue(any("mapped Cuba => CUB" in item for item in evidence))
        self.assertEqual(provenance, ["https://en.wikipedia.org/wiki/1928_Summer_Olympics"])

    @patch("domains.gaia_ops.backend._fetch_document_with_pdf")
    @patch("domains.gaia_ops.backend._search_documents_for_title")
    def test_paper_compare_ops_computes_numeric_difference_between_titles(
        self,
        mock_search_title: Any,
        mock_fetch_pdf: Any,
    ) -> None:
        def _search_side_effect(title: str, *args: Any, **kwargs: Any) -> list[dict[str, str]]:
            return [{"title": title, "snippet": "Journal article", "url": f"https://example.com/{title.replace(' ', '_')}.pdf", "text": ""}]

        def _fetch_side_effect(url: str) -> dict[str, str]:
            if "Paper_A" in url:
                return {"text": "", "pdf_text": "Paper A reports a measured time span of 12.5 milliseconds."}
            return {"text": "", "pdf_text": "Paper B reports a measured time span of 9.0 milliseconds."}

        mock_search_title.side_effect = _search_side_effect
        mock_fetch_pdf.side_effect = _fetch_side_effect

        answer, evidence, provenance = _solve_paper_compare_ops(
            'What is the difference in measured time span between the papers "Paper A" and "Paper B"?'
        )

        self.assertEqual(answer, "3.5")
        self.assertTrue(any("difference between Paper A=12.5 and Paper B=9.0 => 3.5" in item for item in evidence))
        self.assertEqual(len(provenance), 2)

    @patch("domains.gaia_ops.backend._fetch_document_with_pdf")
    @patch("domains.gaia_ops.backend._search_documents_for_title")
    def test_paper_numeric_lookup_passes_anchor_prompt_to_title_search(
        self,
        mock_search_title: Any,
        mock_fetch_pdf: Any,
    ) -> None:
        prompt = 'In Valentina Re’s contribution to the 2017 book "World Building: Transmedia, Fans, Industries", what horror movie does the author cite?'
        mock_search_title.return_value = [
            {
                "title": "World Building: Transmedia, Fans, Industries",
                "snippet": "",
                "url": "https://example.com/world-building.pdf",
                "text": "",
            }
        ]
        mock_fetch_pdf.return_value = {"text": "", "pdf_text": "A Nightmare on Elm Street popularized metalepsis."}

        _solve_paper_numeric_lookup(prompt)

        self.assertEqual(mock_search_title.call_args.kwargs.get("anchor_prompt"), prompt)

    @patch("domains.gaia_ops.backend._fetch_document_with_pdf")
    @patch("domains.gaia_ops.backend._search_documents_for_title")
    @patch("domains.gaia_ops.backend._search_documents_from_prompt")
    def test_paper_compare_ops_computes_integer_rounded_percentage_from_author_year_queries(
        self,
        mock_search_prompt: Any,
        mock_search_title: Any,
        mock_fetch_pdf: Any,
    ) -> None:
        def _search_side_effect(query: str, *args: Any, **kwargs: Any) -> list[dict[str, str]]:
            return [{"title": query, "snippet": "paper", "url": f"https://example.com/{query.replace(' ', '_')}.pdf", "text": ""}]

        def _fetch_side_effect(url: str) -> dict[str, str]:
            if "Omar" in url:
                return {"text": "", "pdf_text": "The total length of the harlequin shrimp was 20 mm."}
            return {"text": "", "pdf_text": "The sea star fed to the shrimp measured 5 mm."}

        mock_search_prompt.side_effect = _search_side_effect
        mock_search_title.side_effect = _search_side_effect
        mock_fetch_pdf.side_effect = _fetch_side_effect

        answer, evidence, provenance = _solve_paper_compare_ops(
            "What integer-rounded percentage of the total length of the harlequin shrimp recorded in Omar Valencia-Mendez 2017 paper was the sea star fed to the same type of shrimp in G. Curt Fiedler's 2002 paper?"
        )

        self.assertEqual(answer, "25")
        self.assertTrue(any("percentage" in item for item in evidence))
        self.assertEqual(len(provenance), 2)

    @patch("domains.gaia_ops.backend._youtube_transcript_segments")
    def test_video_transcript_ops_extracts_command_from_timestamp_window(self, mock_segments: Any) -> None:
        mock_segments.return_value = [
            {"start": 28.0, "end": 33.0, "text": "Now click Command Palette to open the menu."},
            {"start": 33.0, "end": 36.0, "text": "Then choose the extension."},
        ]

        answer, evidence, provenance = _solve_video_transcript_ops(
            "In the video https://www.youtube.com/watch?v=demo123, what command is clicked at 30 seconds?"
        )

        self.assertEqual(answer, "Command Palette")
        self.assertTrue(any("transcript answer=Command Palette" in item for item in evidence))
        self.assertIn("youtube:transcript", provenance)

    @patch("domains.gaia_ops.backend._youtube_transcript_segments")
    def test_video_transcript_ops_extracts_response_after_question(self, mock_segments: Any) -> None:
        mock_segments.return_value = [
            {"start": 10.0, "end": 12.0, "text": "Isn't that hot?"},
            {"start": 12.0, "end": 15.0, "text": "Indeed it is."},
        ]

        answer, evidence, provenance = _solve_video_transcript_ops(
            'Examine the video at https://www.youtube.com/watch?v=demo456. What does Teal\'c say in response to the question "Isn\'t that hot?"'
        )

        self.assertEqual(answer, "Indeed it is")
        self.assertTrue(any("transcript answer=Indeed it is" in item for item in evidence))
        self.assertIn("youtube:transcript", provenance)

    @patch("domains.gaia_ops.backend._youtube_transcript_segments")
    def test_video_transcript_ops_counts_letter_occurrences_in_transcript_phrase(self, mock_segments: Any) -> None:
        mock_segments.return_value = [
            {"start": 28.0, "end": 34.0, "text": '"RED EEL"'},
        ]

        answer, evidence, provenance = _solve_video_transcript_ops(
            'Thirty seconds into the first episode, a phrase is shown on the screen in white letters on a red background. How many times does the letter "E" appear in this phrase? https://www.youtube.com/watch?v=demo789'
        )

        self.assertEqual(answer, "3")
        self.assertTrue(any("transcript answer=3" in item for item in evidence))
        self.assertIn("youtube:transcript", provenance)

    @patch("domains.gaia_ops.backend._public_reference_search_documents")
    @patch("domains.gaia_ops.backend._discover_video_url")
    def test_video_transcript_ops_falls_back_to_replit_page_structure_when_no_video_url(
        self,
        mock_discover_video_url: Any,
        mock_search_documents: Any,
    ) -> None:
        mock_discover_video_url.return_value = ""
        mock_search_documents.return_value = [
            {
                "title": "Zero Setup VSCode Intelligence - Replit Blog",
                "url": "https://blog.replit.com/intel",
                "text": "<h2>Autocomplete and signatures</h2><h2>Formatting</h2>",
            }
        ]

        answer, evidence, provenance = _solve_video_transcript_ops(
            "In the 2018 VSCode blog post on replit.com, what was the command they clicked on in the last video to remove extra lines?"
        )

        self.assertEqual(answer, "Format Document")
        self.assertTrue(any("video fallback via page structure" in item for item in evidence))
        self.assertEqual(provenance, ["https://blog.replit.com/intel"])

    @patch("domains.gaia_ops.backend._public_reference_search_documents")
    def test_solve_question_replit_video_prompt_sets_candidate_answer_via_video_fallback(self, mock_search_documents: Any) -> None:
        backend = GaiaOpsReasoningDomain()
        task = ReasoningTask(
            task_id="replit_probe",
            domain="gaia_json_reasoning",
            prompt="In the 2018 VSCode blog post on replit.com, what was the command they clicked on in the last video to remove extra lines?",
            answer="",
            goal="Return the correct final answer",
            meta={},
        )
        state = backend.make_state(task)
        plan_result = plan_question(state.problem_text, state)
        state.metadata.update(plan_result["payload"]["state_metadata"])
        mock_search_documents.return_value = [
            {
                "title": "Zero Setup VSCode Intelligence - Replit Blog",
                "url": "https://blog.replit.com/intel",
                "text": "<h2>Autocomplete and signatures</h2><h2>Formatting</h2>",
            }
        ]

        result = solve_question(state.problem_text, state)

        self.assertTrue(result["ok"])
        self.assertEqual(result["payload"]["candidate_answer"], "Format Document")
        self.assertGreaterEqual(result["payload"]["state_metadata"]["answer_confidence"], 0.72)

    @patch("domains.gaia_ops.backend._solve_thinking_machine_prediction", side_effect=AssertionError("legacy helper should not run"))
    @patch("domains.gaia_ops.backend._solve_youtube_bird_species_count", side_effect=AssertionError("legacy helper should not run"))
    @patch("domains.gaia_ops.backend._youtube_video_metadata")
    @patch("domains.gaia_ops.backend._fetch_search_documents")
    @patch("domains.gaia_ops.backend._youtube_transcript_segments")
    def test_video_transcript_ops_uses_generic_document_scalar_without_legacy_story_helper(
        self,
        mock_segments: Any,
        mock_search_docs: Any,
        mock_video_metadata: Any,
        _legacy_birds: Any,
        _legacy_prediction: Any,
    ) -> None:
        mock_segments.return_value = []
        mock_video_metadata.return_value = {}
        mock_search_docs.return_value = [
            {
                "url": "https://example.com/companion-article",
                "title": "Companion article",
                "snippet": "The highest number of bird species seen simultaneously was 7.",
                "text": "Observers reported the highest number of bird species on camera simultaneously was 7.",
            }
        ]

        answer, evidence, provenance = _solve_video_transcript_ops(
            "In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?"
        )

        self.assertEqual(answer, "7")
        self.assertTrue(any("video_document_scalar" in item for item in evidence))
        self.assertIn("https://example.com/companion-article", provenance)

    @patch("domains.gaia_ops.backend._solve_thinking_machine_prediction", side_effect=AssertionError("legacy helper should not run"))
    @patch("domains.gaia_ops.backend._solve_youtube_bird_species_count", side_effect=AssertionError("legacy helper should not run"))
    @patch("domains.gaia_ops.backend._youtube_video_metadata")
    @patch("domains.gaia_ops.backend._fetch_search_documents")
    @patch("domains.gaia_ops.backend._youtube_transcript_segments")
    def test_video_transcript_ops_counts_species_from_fused_evidence_without_legacy_helper(
        self,
        mock_segments: Any,
        mock_search_docs: Any,
        mock_video_metadata: Any,
        _legacy_birds: Any,
        _legacy_prediction: Any,
    ) -> None:
        mock_segments.return_value = [
            {"start": 1.0, "end": 4.0, "text": "An emperor penguin moves past a giant petrel."},
        ]
        mock_video_metadata.return_value = {
            "title": "Antarctic birds on camera",
            "description": "Features an Adélie penguin near the colony.",
        }
        mock_search_docs.return_value = [
            {
                "url": "https://example.com/birds",
                "title": "Bird guide",
                "snippet": "Gentoo penguin joins the group.",
                "text": "Observers later note a gentoo penguin with the emperor penguin and giant petrel.",
            }
        ]

        answer, evidence, provenance = _solve_video_transcript_ops(
            "In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?"
        )

        self.assertEqual(answer, "4")
        self.assertTrue(any("video species detected=" in item for item in evidence))
        self.assertIn("https://example.com/birds", provenance)

    @patch("domains.gaia_ops.backend._solve_thinking_machine_prediction", side_effect=AssertionError("legacy helper should not run"))
    @patch("domains.gaia_ops.backend._solve_youtube_bird_species_count", side_effect=AssertionError("legacy helper should not run"))
    @patch("domains.gaia_ops.backend._youtube_video_metadata")
    @patch("domains.gaia_ops.backend._fetch_search_documents")
    @patch("domains.gaia_ops.backend._youtube_transcript_segments")
    def test_video_transcript_ops_uses_generic_person_evidence_graph_without_legacy_helper(
        self,
        mock_segments: Any,
        mock_search_docs: Any,
        mock_video_metadata: Any,
        _legacy_birds: Any,
        _legacy_prediction: Any,
    ) -> None:
        mock_segments.return_value = [
            {"start": 10.0, "end": 18.0, "text": "Claude Shannon says thinking machines will handle routine work."},
        ]
        mock_video_metadata.return_value = {
            "title": "The Thinking Machine",
            "description": "A scientist predicts the future of AI and robots.",
        }
        mock_search_docs.return_value = [
            {
                "url": "https://example.com/thinking-machine",
                "title": "The Thinking Machine and Claude Shannon",
                "snippet": "Claude Shannon is the scientist making the prediction about the future of AI.",
                "text": "In the film The Thinking Machine, Claude Shannon is explicitly identified as the scientist making the prediction about the future of AI and robots.",
            }
        ]

        answer, evidence, provenance = _solve_video_transcript_ops(
            "In the video https://www.youtube.com/watch?v=demo999, which scientist is predicting the future of AI and robots?"
        )

        self.assertEqual(answer, "Claude Shannon")
        self.assertTrue(any("video best person=Claude Shannon" in item for item in evidence))
        self.assertTrue(provenance)

    @patch("domains.gaia_ops.backend._solve_thinking_machine_prediction", side_effect=AssertionError("legacy helper should not run"))
    @patch("domains.gaia_ops.backend._solve_youtube_bird_species_count", side_effect=AssertionError("legacy helper should not run"))
    @patch("domains.gaia_ops.backend._youtube_video_metadata")
    @patch("domains.gaia_ops.backend._fetch_search_documents")
    @patch("domains.gaia_ops.backend._youtube_transcript_segments")
    def test_video_transcript_ops_ignores_low_value_youtube_docs_for_person_lookup(
        self,
        mock_segments: Any,
        mock_search_docs: Any,
        mock_video_metadata: Any,
        _legacy_birds: Any,
        _legacy_prediction: Any,
    ) -> None:
        mock_segments.return_value = []
        mock_video_metadata.return_value = {
            "title": "The Thinking Machine",
            "description": "Documentary about scientists predicting AI.",
        }
        mock_search_docs.return_value = [
            {
                "url": "https://www.youtube.com/watch?v=deadbeef",
                "title": "About Press Copyright Contact us Creators Advertise Developers",
                "snippet": "Before you continue to YouTube",
                "text": "Privacy Policy Terms of Service Accept all Reject all",
            },
            {
                "url": "https://example.com/thinking-machine",
                "title": "The Thinking Machine and Claude Shannon",
                "snippet": "Claude Shannon is the scientist making the prediction about the future of AI.",
                "text": "Claude Shannon is identified as the scientist predicting the future of AI and robots.",
            },
        ]

        answer, evidence, provenance = _solve_video_transcript_ops(
            "In the video https://www.youtube.com/watch?v=demo999, which scientist is predicting the future of AI and robots?"
        )

        self.assertEqual(answer, "Claude Shannon")
        self.assertTrue(any("video best person=Claude Shannon" in item for item in evidence))
        self.assertEqual(provenance, ["https://example.com/thinking-machine"])

    @patch("domains.gaia_ops.backend._solve_numpy_regression_github_case", side_effect=AssertionError("legacy helper should not run"), create=True)
    @patch("domains.gaia_ops.backend._github_issue_timeline_events")
    @patch("domains.gaia_ops.backend._github_search_issues")
    @patch("domains.gaia_ops.backend._fetch_search_documents")
    def test_github_public_artifact_ops_uses_generic_issue_timeline_without_legacy_named_helper(
        self,
        mock_search_docs: Any,
        mock_search_issues: Any,
        mock_timeline: Any,
        _legacy_solver: Any,
    ) -> None:
        mock_search_docs.return_value = [
            {
                "url": "https://github.com/numpy/numpy/issues/1841",
                "title": "numpy/numpy issue 1841",
                "snippet": "Regression issue with label component: numpy.polynomial",
                "text": "Issue in repo numpy/numpy",
            }
        ]
        mock_search_issues.return_value = [
            {
                "number": 1841,
                "html_url": "https://github.com/numpy/numpy/issues/1841",
                "closed_at": "2019-05-10T00:00:00Z",
            }
        ]
        mock_timeline.return_value = [
            {
                "event": "labeled",
                "created_at": "2019-05-04T12:00:00Z",
                "label": {"name": "regression"},
            }
        ]

        answer, evidence, provenance = _solve_github_public_artifact_ops(
            "According to GitHub, what date was the earliest closed issue with the label component: numpy.polynomial labeled as regression? Answer in MM/DD/YY."
        )

        self.assertEqual(answer, "05/04/19")
        self.assertTrue(any("generic_github_issue_event" in item for item in evidence))
        self.assertEqual(provenance, ["https://github.com/numpy/numpy/issues/1841"])

    @patch("domains.gaia_ops.backend._wayback_snapshot_html")
    @patch("domains.gaia_ops.backend._search_documents_from_prompt")
    def test_web_archive_ops_extracts_removed_menu_item(self, mock_search_docs: Any, mock_snapshot_html: Any) -> None:
        mock_search_docs.return_value = [
            {
                "title": "Virtue menu",
                "url": "https://example.com/menu",
                "text": "Menu archive",
                "html_text": "<html><body>Menu archive</body></html>",
            }
        ]
        mock_snapshot_html.side_effect = [
            "<html><body><ul><li>Carrot soup</li><li>Winter salad</li></ul></body></html>",
            "<html><body><ul><li>Winter salad</li></ul></body></html>",
        ]

        answer, evidence, provenance = _solve_web_archive_ops(
            "Using the Wayback Machine, which menu item appeared on the restaurant website in March 2020 but no longer appeared in July 2020?"
        )

        self.assertEqual(answer, "Carrot soup")
        self.assertTrue(any("removed item" in item for item in evidence))
        self.assertEqual(provenance, ["https://example.com/menu", "wayback:diff"])

    @patch("domains.gaia_ops.backend._wayback_snapshot_html")
    @patch("domains.gaia_ops.backend._search_documents_from_prompt")
    def test_web_archive_ops_extracts_deleted_word_between_versions(
        self, mock_search_docs: Any, mock_snapshot_html: Any
    ) -> None:
        mock_search_docs.return_value = [
            {
                "title": "Amendment page",
                "url": "https://example.com/amendment",
                "text": "Amendment archive",
                "html_text": "<html><body>Amendment archive</body></html>",
            }
        ]
        mock_snapshot_html.side_effect = [
            "<html><body>The amendment guarantees liberty and justice for all.</body></html>",
            "<html><body>The amendment guarantees justice for all.</body></html>",
        ]

        answer, evidence, provenance = _solve_web_archive_ops(
            "Using the Wayback Machine, what word was deleted in the last amendment between January 2018 and January 2019?"
        )

        self.assertEqual(answer, "liberty")
        self.assertTrue(any("deleted word" in item for item in evidence))
        self.assertEqual(provenance, ["https://example.com/amendment", "wayback:diff"])

    @patch("domains.gaia_ops.backend._easyocr_text_lines")
    def test_image_vision_ops_extracts_fraction_list(self, mock_lines: Any) -> None:
        mock_lines.return_value = ["1/2 3/4", "noise"]

        answer, evidence, provenance = _solve_image_vision_ops(
            "List the fractions shown in the image.",
            [Path("fractions.png")],
        )

        self.assertEqual(answer, "1/2,3/4")
        self.assertTrue(any("fractions from fractions.png" in item for item in evidence))
        self.assertEqual(provenance, ["image:fractions.png"])

    @patch("domains.gaia_ops.backend._easyocr_text_lines_with_variants")
    def test_image_vision_ops_extracts_latest_year(self, mock_lines: Any) -> None:
        mock_lines.return_value = ["1894", "2003", "1998"]

        answer, evidence, provenance = _solve_image_vision_ops(
            "In the attached image, what is the latest chronological year that appears?",
            [Path("poster.jpg")],
        )

        self.assertEqual(answer, "2003")
        self.assertTrue(any("years from poster.jpg" in item for item in evidence))
        self.assertEqual(provenance, ["image:poster.jpg"])

    def test_image_code_solver_reconstructs_archive_url_and_sums_sorted_positions(self) -> None:
        observation = {
            "lines": [
                "archive_prefix",
                "'https / Iweb.archive_",
                "org/web/20230609112831/'",
                "url_indices",
                "[33,4,8,9,10,14,17,18,19,20,21,22,24,23,0,26,27,28,5,30,31,32,2]",
            ],
            "variant_lines": [],
        }

        answer, evidence = _solve_embedded_sorting_code_image(
            "The attached image contains a Python script. Run the Python code against an array of strings, listed below. "
            "The output of the Python script will be a URL containing C++ source code. Compile and run this C++ code "
            "against the array [35, 12, 8, 99, 21, 5] and return the sum of the third and fifth integers in the sorted list.\n\n"
            "arr = ['_alg', 'ghi', 'C++', 'jkl', 'tps', '/Q', 'pqr', 'stu', ':', '//', 'rose', 'vwx', 'yz1', '234', 'tta', "
            "'567', '890', 'cod', 'e.', 'or', 'g/', 'wiki', '/', 'ing', 'sort', 'abc' , 'or', 'it', 'hms', 'mno' , 'uic', 'ksort', '#', 'ht' ]",
            observation,
        )

        self.assertEqual(answer, "47")
        self.assertTrue(any("web.archive.org" in item for item in evidence))
        self.assertTrue(any("sorted numbers=[5, 8, 12, 21, 35, 99]" in item for item in evidence))

    def test_storage_plan_upgrade_math_chooses_minimum_fitting_tier(self) -> None:
        boxes = [
            (((90, 54, 230, 84)), "Standard"),
            (((412, 54, 482, 84)), "Plus"),
            (((672, 51, 809, 87)), "Premium"),
            (((46, 102, 192, 132)), "$9.99/month"),
            (((334, 100, 492, 128)), "$19.99/month"),
            (((622, 102, 782, 130)), "$39.99/month"),
            (((74, 150, 200, 178)), "2 TB storage"),
            (((362, 146, 501, 177)), "10 TB storage"),
            (((648, 144, 787, 175)), "50 TB storage"),
        ]

        answer, evidence = _solve_storage_plan_upgrade_math(
            "I have the Standard plan in the image below, and I just uploaded 60 equally sized files and got a message that I'm 100GB over the limit. "
            "I have 980 more files of the same size to upload. What is the average additional cost per file in dollar that goes over my current plan limit "
            "rounded to the nearest cent if I have to upgrade to the minimum possible plan to store them all? Answer with the following format: x.xx",
            boxes,
        )

        self.assertEqual(answer, "0.03")
        self.assertTrue(any("target plan=Premium" in item for item in evidence))

    def test_infer_fraction_pair_prefers_reducible_candidate(self) -> None:
        pair = _infer_fraction_pair_from_ocr_sequences(
            2,
            [
                [2, 1, 60],
                [2, 4, 60],
                [2, 60],
                [2, 68],
            ],
        )

        self.assertEqual(pair, (4, 60))

    def test_best_quiz_arithmetic_match_finds_nearest_exact_fraction(self) -> None:
        match = _best_quiz_arithmetic_match(("20", "35", "18", "47"), "522/1645")

        self.assertIsNotNone(match)
        self.assertEqual(match["operation"], "mul")
        self.assertEqual(match["numbers"], ("29", "35", "18", "47"))
        self.assertEqual(match["score"], 1)

    def test_quiz_conversion_match_handles_improper_and_mixed_rows(self) -> None:
        improper_to_mixed = _best_quiz_improper_to_mixed_match("85", "32", "2 21/32")
        mixed_to_improper = _best_quiz_mixed_to_improper_match("32", "8", "9", "293/9")

        self.assertIsNotNone(improper_to_mixed)
        self.assertEqual(improper_to_mixed["score"], 0)
        self.assertIsNotNone(mixed_to_improper)
        self.assertEqual(mixed_to_improper["mixed"], "32 5/9")
        self.assertEqual(mixed_to_improper["score"], 1)

    def test_butterfat_nutrition_solver_uses_noisy_label_tokens(self) -> None:
        answer, evidence = _solve_butterfat_nutrition_image(
            "If this whole pint is made up of ice cream, how many percent above or below the US federal standards for butterfat content is it when using the standards as reported by Wikipedia in 2020?",
            {
                "variant_lines": [
                    "Serving size 2/3 cup (96g)",
                    "Total Fat 814%",
                    "Calories 2224",
                ]
            },
        )

        self.assertEqual(answer, "+4.6")
        self.assertTrue(any("fat_grams=14" in item for item in evidence))

    def test_green_step_area_helpers_snap_segments_to_expected_area(self) -> None:
        x_units = _round_segments_to_total([55.5, 62.5, 58.0, 21.0, 59.0], 10.0)
        y_units = [max(0.5, round((value / (40.5 / 1.5)) * 2.0) / 2.0) for value in [59.5, 68.5, 40.5, 59.0, 82.5]]

        self.assertEqual(x_units, [2.0, 2.5, 2.0, 1.0, 2.5])
        self.assertEqual(y_units, [2.0, 2.5, 1.5, 2.0, 3.0])
        self.assertEqual(int(round(_green_step_area(x_units, y_units))), 39)

    @patch("domains.gaia_ops.backend._easyocr_text_lines")
    def test_easyocr_text_lines_with_variants_tries_multiple_preprocessed_images(self, mock_lines: Any) -> None:
        mock_lines.side_effect = [[], ["1927"], ["1927"], [], []]
        image_path = Path(".tmp-benchmarks/gaia/tests-ocr-variants.png")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 12), color="white").save(image_path)

        lines = _easyocr_text_lines_with_variants(image_path)

        self.assertEqual(lines, ["1927"])
        self.assertGreaterEqual(mock_lines.call_count, 2)

    @patch("domains.gaia_ops.backend._solve_github_contributor_name_match", side_effect=AssertionError("legacy helper should not run"))
    @patch("domains.gaia_ops.backend._fetch_search_documents")
    def test_github_public_artifact_ops_routes_contributor_match_structurally(
        self, mock_search_docs: Any, _legacy_solver: Any
    ) -> None:
        def _fake_search(query: str, **_: Any) -> list[dict[str, str]]:
            if "same name as" in query.lower() or "github" in query.lower():
                return [
                    {
                        "url": "https://github.com/opencv/opencv/pull/1",
                        "title": "OpenCV PR by Zhao Ziyang",
                        "snippet": "Contributor Zhao Ziyang added Mask-RCNN support.",
                        "text": "Contributor Zhao Ziyang added Mask-RCNN support.",
                    }
                ]
            if "former chinese head of government" in query.lower():
                return [
                    {
                        "url": "https://en.wikipedia.org/wiki/Zhao_Ziyang",
                        "title": "Zhao Ziyang - former Chinese head of government",
                        "snippet": "Zhao Ziyang served as premier of China.",
                        "text": "Zhao Ziyang served as premier of China.",
                    }
                ]
            return []

        mock_search_docs.side_effect = _fake_search

        answer, evidence, provenance = _solve_github_public_artifact_ops(
            "Which contributor to the version of OpenCV where support was added for the Mask-RCNN model has the same name as a former Chinese head of government when the names are transliterated to the Latin alphabet?"
        )

        self.assertEqual(answer, "Zhao Ziyang")
        self.assertTrue(any("generic_github_contributor_match" in item for item in evidence))
        self.assertEqual(
            provenance,
            ["https://github.com/opencv/opencv/pull/1", "https://en.wikipedia.org/wiki/Zhao_Ziyang"],
        )

    @patch("domains.gaia_ops.backend._solve_generic_public_reference")
    def test_solve_question_generic_public_reference_requires_stronger_confidence_before_candidate_answer(
        self, mock_solve: Any
    ) -> None:
        mock_solve.return_value = ("16", ["row count from section"], ["wikipedia:Mercedes Sosa"])
        state = SimpleNamespace(
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": [],
                "question_plan": {},
                "target_file": "",
                "candidate_files": [],
                "inspected_files": [],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
            },
            problem_text="How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)?",
        )

        result = solve_question(state.problem_text, state)

        self.assertFalse(result["ok"])
        self.assertIn("quality checks", result["result"])

    @patch("domains.gaia_ops.backend._solve_generic_public_reference")
    def test_solve_question_generic_public_reference_exposes_candidate_answer_when_evidence_is_strong(
        self, mock_solve: Any
    ) -> None:
        mock_solve.return_value = (
            "3",
            ["title=Mercedes Sosa discography", "year range count 2000-2009 => 3"],
            ["wikipedia:Mercedes Sosa"],
        )
        state = SimpleNamespace(
            metadata={
                "workspace_dir": str(Path.cwd()),
                "workspace_files": [],
                "question_plan": {},
                "target_file": "",
                "candidate_files": [],
                "inspected_files": [],
                "benchmark_assistance_mode": "unassisted",
                "oracle_hints_enabled": False,
            },
            problem_text="How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)?",
        )

        result = solve_question(state.problem_text, state)

        self.assertTrue(result["ok"])
        self.assertNotIn("solved", result)
        self.assertEqual(result["payload"]["candidate_answer"], "3")
        self.assertEqual(result["payload"]["state_metadata"]["candidate_answer"], "3")
        self.assertGreaterEqual(result["payload"]["state_metadata"]["answer_confidence"], 0.72)
        self.assertEqual(result["payload"]["state_metadata"]["reasoning_schema"]["source_family"], "public_reference")
        self.assertEqual(result["payload"]["state_metadata"]["augmentation_layer"]["mode"], "trillion_structural")
        self.assertEqual(
            result["payload"]["state_metadata"]["task_algebra"]["equation"],
            "time x source x operator x contract x rival",
        )
        self.assertEqual(
            result["payload"]["state_metadata"]["internal_role_machine"]["roles"],
            "framer -> retriever -> resolver -> judge -> closer",
        )
        self.assertTrue(result["payload"]["state_metadata"]["answer_self_check"]["accepted"])

    def test_fallback_repairs_do_not_answer_low_confidence_generic_public_reference_candidates(self) -> None:
        backend = GaiaOpsReasoningDomain()
        state = SimpleNamespace(
            problem_text="How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)?",
            tool_history=[
                {"tool": "plan_question"},
                {"tool": "list_files"},
                {"tool": "inspect_file"},
                {"tool": "solve_question"},
            ],
            metadata={
                "question_plan": {},
                "candidate_answer": "16",
                "answer_confidence": 0.68,
                "answer_mode": "generic_public_reference",
                "target_file": "",
                "candidate_files": [],
                "inspected_files": [],
            },
            final_answer="",
            derived_facts=[],
            obligations=[],
        )

        action = backend.fallback_repairs(cast(ReasoningState, state))[0]

        self.assertEqual(action.type, ActionType.BACKTRACK)

    def test_fallback_repairs_answers_high_confidence_generic_public_reference_candidates_even_with_failed_self_check(self) -> None:
        backend = GaiaOpsReasoningDomain()
        state = SimpleNamespace(
            problem_text="How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)?",
            tool_history=[
                {"tool": "plan_question"},
                {"tool": "list_files"},
                {"tool": "inspect_file"},
                {"tool": "solve_question"},
            ],
            metadata={
                "question_plan": {},
                "candidate_answer": "16",
                "answer_confidence": 0.92,
                "answer_mode": "generic_public_reference",
                "target_file": "",
                "candidate_files": [],
                "inspected_files": [],
                "answer_self_check": {
                    "accepted": False,
                    "support": 0.18,
                    "notes": ["temporal provenance weak"],
                },
            }
        )

        action = backend.fallback_repairs(cast(ReasoningState, state))[0]

        self.assertEqual(action.type, ActionType.ANSWER)
