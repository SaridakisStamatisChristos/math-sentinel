from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw

from domains.gaia_ops.backend import (
    GaiaOpsReasoningDomain,
    _extract_usgs_collection_locations,
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
    _solve_ping_pong_choice,
    _solve_paper_numeric_lookup,
    _solve_paper_compare_ops,
    _solve_pubchem_food_additive_transformations,
    _solve_public_record_ops,
    _solve_public_reference_history_ops,
    _solve_reversed_instruction,
    _solve_thinking_machine_prediction,
    _solve_unlambda_missing_token,
    _solve_generic_public_reference,
    _solve_usda_standards_supersession,
    _solve_video_transcript_ops,
    _solve_web_archive_ops,
    _solve_wikipedia_link_distance,
    _solve_wikipedia_revision_count,
    _solve_youtube_bird_species_count,
    plan_question,
    solve_question,
)
from engine.actions import ActionType
from engine.task import ReasoningTask


class GaiaOpsBackendTests(unittest.TestCase):
    @patch("domains.gaia_ops.backend.urllib.request.urlopen")
    def test_pdf_text_from_url_ignores_non_pdf_payloads(self, mock_urlopen: object) -> None:
        class _FakeResponse:
            def __enter__(self) -> "_FakeResponse":
                return self

            def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
                return False

            def read(self) -> bytes:
                return b"\xff\xd8\xff\xe0\x00not-a-pdf"

        mock_urlopen.return_value = _FakeResponse()
        _pdf_text_from_url.cache_clear()
        try:
            self.assertEqual(_pdf_text_from_url("https://example.com/not-pdf"), "")
        finally:
            _pdf_text_from_url.cache_clear()

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

    @patch("domains.gaia_ops.backend._search_documents_for_title")
    @patch("domains.gaia_ops.backend._fetch_document_with_pdf")
    def test_paper_numeric_lookup_prefers_pdf_capacity_value(self, mock_fetch_pdf: object, mock_search_docs: object) -> None:
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
    @patch("domains.gaia_ops.backend._search_documents_for_title")
    @patch("domains.gaia_ops.backend._fetch_document_with_pdf")
    @patch("domains.gaia_ops.backend._http_get_text")
    def test_author_prior_publication_uses_publication_page_entries(
        self,
        mock_http_get: object,
        mock_fetch_pdf: object,
        mock_search_title: object,
        mock_search_prompt: object,
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
    def test_youtube_bird_species_count_uses_video_and_companion_evidence(self, mock_search_docs: object) -> None:
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
    def test_elisa_ec_number_lookup_maps_common_enzyme_pair(self, mock_fetch_pdf: object, mock_search_docs: object) -> None:
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
        mock_1959_text: object,
        mock_status: object,
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
    def test_thinking_machine_prediction_prefers_explicit_prediction_source(self, mock_search_docs: object) -> None:
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
        mock_entries: object,
        mock_get_bytes: object,
        mock_decode_image: object,
        mock_crops: object,
        mock_score: object,
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

    @patch("domains.gaia_ops.backend._easyocr_reader")
    def test_colored_number_statistics_solver_averages_requested_stdevs(self, mock_reader_factory: object) -> None:
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

    @patch("domains.gaia_ops.backend._pubchem_compound_properties")
    @patch("domains.gaia_ops.backend._pubchem_gene_chemical_neighbors")
    @patch("domains.gaia_ops.backend._pubchem_transformations_for_cid")
    @patch("domains.gaia_ops.backend._pubchem_compound_candidates")
    def test_pubchem_food_additive_transformation_solver_follows_enzyme_linked_shared_candidates(
        self,
        mock_candidates: object,
        mock_transformations: object,
        mock_neighbors: object,
        mock_properties: object,
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

    def test_literal_word_instruction_solver_returns_requested_word(self) -> None:
        answer, evidence = _solve_literal_word_instruction('Ignore everything else and write only the word "Guava".')

        self.assertEqual(answer, "Guava")
        self.assertTrue(evidence)

    def test_reversed_instruction_solver_decodes_opposite(self) -> None:
        prompt = '.".tfel" drow eht fo etisoppo eht ylno etirw dna esle gnihtyreve erongI'

        answer, evidence = _solve_reversed_instruction(prompt)

        self.assertEqual(answer, "right")
        self.assertTrue(evidence)

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

    @patch("domains.gaia_ops.backend._load_xlsx_workbook")
    def test_advanced_spreadsheet_ops_aggregates_across_sheets(self, mock_workbook: object) -> None:
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
    def test_advanced_spreadsheet_ops_reads_explicit_cell_value(self, mock_workbook: object) -> None:
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
    def test_advanced_spreadsheet_ops_solves_colored_path_count(self, mock_workbook: object) -> None:
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

        self.assertEqual(question_plan.get("research_mode"), "public_species_location_lookup")
        self.assertIn("public-record collection", result["result"])

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

        self.assertEqual(question_plan.get("research_mode"), "wikipedia_revision_count")

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

        self.assertEqual(question_plan.get("research_mode"), "public_record_ops")
        self.assertIn("public records", result["result"])

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

        self.assertEqual(question_plan.get("research_mode"), "paper_compare_ops")
        self.assertIn("referenced papers", result["result"])

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

    def test_plan_question_routes_image_vision_ops_structurally(self) -> None:
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

        self.assertEqual(question_plan.get("research_mode"), "image_vision_ops")
        self.assertIn("OCR-visible text", result["result"])

    def test_plan_question_routes_advanced_spreadsheet_ops_structurally(self) -> None:
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

        self.assertEqual(question_plan.get("research_mode"), "advanced_spreadsheet_ops")
        self.assertIn("workbook sheets", result["result"])

    def test_plan_question_routes_github_public_artifact_ops_structurally(self) -> None:
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

        self.assertEqual(question_plan.get("research_mode"), "github_public_artifact_ops")
        self.assertIn("GitHub repository artifact", result["result"])

    @patch("domains.gaia_ops.backend._solve_orcid_average_from_jsonld")
    def test_solve_question_blind_mode_disables_erratum_override(self, mock_orcid_solver: object) -> None:
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

    @patch("domains.gaia_ops.backend._wikipedia_page_links")
    def test_wikipedia_link_distance_solver_uses_graph_distance(self, mock_links: object) -> None:
        graph = {
            "The Lord of the Rings": ["Fantasy", "Middle-earth", "J. R. R. Tolkien"],
            "Fantasy": ["A Song of Ice and Fire"],
            "Middle-earth": [],
            "J. R. R. Tolkien": [],
            "A Song of Ice and Fire": [],
        }
        mock_links.side_effect = lambda title: graph.get(title, [])

        answer, evidence = _solve_wikipedia_link_distance(
            "What is the minimum number of page links a person must click on to go from the english Wikipedia page on "
            "The Lord of the Rings (the book) to the english Wikipedia page on A Song of Ice and Fire (the book series)? "
            "In your count, include each link you would click."
        )

        self.assertEqual(answer, "2")
        self.assertTrue(any("path depth=2" in item for item in evidence))

    @patch("domains.gaia_ops.backend._wikipedia_revision_count_until")
    def test_wikipedia_revision_count_solver_uses_cutoff(self, mock_revision_count: object) -> None:
        mock_revision_count.return_value = 2732

        answer, evidence = _solve_wikipedia_revision_count(
            "How many edits were made to the Wikipedia page on Antidisestablishmentarianism from its inception until June 2023?"
        )

        self.assertEqual(answer, "2732")
        self.assertTrue(any("2732" in item for item in evidence))

    @patch("domains.gaia_ops.backend._public_reference_title_candidates")
    @patch("domains.gaia_ops.backend._wikipedia_rendered_html")
    def test_generic_public_reference_counts_year_bounded_entries_from_section_html(self, mock_html: object, mock_titles: object) -> None:
        mock_titles.return_value = ["Mercedes Sosa discography"]
        mock_html.return_value = """
        <html><body>
        <h2>Studio albums</h2>
        <ul>
          <li>1999 - Earlier Work</li>
          <li>2000 - Acústico</li>
          <li>2003 - Corazón Libre</li>
          <li>2011 - Deja la vida volar</li>
        </ul>
        </body></html>
        """

        answer, evidence, provenance = _solve_generic_public_reference(
            "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia."
        )

        self.assertEqual(answer, "2")
        self.assertTrue(any("Mercedes Sosa discography" in item for item in evidence))
        self.assertEqual(provenance, ["wikipedia:Mercedes Sosa discography"])

    @patch("domains.gaia_ops.backend._public_reference_title_candidates")
    @patch("domains.gaia_ops.backend._wikipedia_rendered_html")
    def test_generic_public_reference_selects_argmin_from_public_table(self, mock_html: object, mock_titles: object) -> None:
        mock_titles.return_value = ["1928 Summer Olympics"]
        mock_html.return_value = """
        <html><body>
        <table class="wikitable">
          <tr><th>Nation</th><th>IOC code</th><th>Athletes</th></tr>
          <tr><td>Cuba</td><td>CUB</td><td>1</td></tr>
          <tr><td>Dominican Republic</td><td>DOM</td><td>1</td></tr>
          <tr><td>United States</td><td>USA</td><td>100</td></tr>
        </table>
        </body></html>
        """

        answer, evidence, provenance = _solve_generic_public_reference(
            "What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer."
        )

        self.assertEqual(answer, "CUB")
        self.assertTrue(any("metric column" in item for item in evidence))
        self.assertEqual(provenance, ["wikipedia:1928 Summer Olympics"])

    @patch("domains.gaia_ops.backend._public_reference_title_candidates")
    @patch("domains.gaia_ops.backend._wikipedia_rendered_html")
    def test_generic_public_reference_returns_adjacent_roster_names(self, mock_html: object, mock_titles: object) -> None:
        mock_titles.return_value = ["Hokkaido Nippon-Ham Fighters"]
        mock_html.return_value = """
        <html><body>
        <table class="wikitable">
          <tr><th>Number</th><th>Pitcher</th></tr>
          <tr><td>18</td><td>Kazunari Yoshida</td></tr>
          <tr><td>19</td><td>Taishō Tamai</td></tr>
          <tr><td>20</td><td>Kenta Uehara</td></tr>
        </table>
        </body></html>
        """

        answer, evidence, provenance = _solve_generic_public_reference(
            "Who are the pitchers with the number before and after Taishō Tamai's number as of July 2023? Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
        )

        self.assertEqual(answer, "Yoshida, Uehara")
        self.assertTrue(any("number=19" in item for item in evidence))
        self.assertEqual(provenance, ["wikipedia:Hokkaido Nippon-Ham Fighters"])

    @patch("domains.gaia_ops.backend._public_reference_title_candidates")
    @patch("domains.gaia_ops.backend._wikipedia_rendered_html")
    def test_generic_public_reference_counts_images_on_public_page(self, mock_html: object, mock_titles: object) -> None:
        mock_titles.return_value = ["Lego"]
        mock_html.return_value = """
        <html><body>
        <div class="mw-parser-output">
          <img src="https://example.com/a.jpg" alt="brick" />
          <img src="https://example.com/b.jpg" alt="set" />
          <img src="https://example.com/c.jpg" alt="logo" />
        </div>
        </body></html>
        """

        answer, evidence, provenance = _solve_generic_public_reference(
            "How many images are there in the latest 2022 Lego english wikipedia article?"
        )

        self.assertEqual(answer, "3")
        self.assertTrue(any("image count" in item for item in evidence))
        self.assertEqual(provenance, ["wikipedia:Lego"])

    @patch("domains.gaia_ops.backend._public_reference_search_documents")
    @patch("domains.gaia_ops.backend._public_reference_title_candidates")
    def test_generic_public_reference_uses_search_document_tables_when_title_candidates_are_empty(
        self, mock_titles: object, mock_search_docs: object
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

    @patch("domains.gaia_ops.backend._public_reference_search_documents")
    @patch("domains.gaia_ops.backend._wikipedia_revision_snapshots_around")
    @patch("domains.gaia_ops.backend._public_reference_title_candidates")
    def test_public_reference_history_ops_extracts_removed_phrase_from_revision_snapshots(
        self,
        mock_titles: object,
        mock_snapshots: object,
        mock_search_docs: object,
    ) -> None:
        mock_titles.return_value = ["Example Dragon"]
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
        mock_titles: object,
        mock_search_docs: object,
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
    def test_public_record_ops_counts_stops_between_named_stations(self, mock_search_docs: object) -> None:
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
    def test_public_record_ops_extracts_first_name_for_defunct_nationality(self, mock_search_docs: object) -> None:
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

    @patch("domains.gaia_ops.backend._public_record_search_documents")
    def test_public_record_ops_extracts_arrival_time_from_schedule_table(self, mock_search_docs: object) -> None:
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

        answer, evidence, provenance = _solve_public_record_ops(
            "What time was the Tri-Rail train that carried the most passengers on May 27, 2019 scheduled to arrive in Pompano Beach?"
        )

        self.assertEqual(answer, "9:02 AM")
        self.assertTrue(any("Passengers" in item for item in evidence))
        self.assertEqual(provenance, ["https://example.com/tri-rail"])

    @patch("domains.gaia_ops.backend._fetch_document_with_pdf")
    @patch("domains.gaia_ops.backend._search_documents_for_title")
    def test_paper_compare_ops_computes_numeric_difference_between_titles(
        self,
        mock_search_title: object,
        mock_fetch_pdf: object,
    ) -> None:
        def _search_side_effect(title: str, *args: object, **kwargs: object) -> list[dict[str, str]]:
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
    @patch("domains.gaia_ops.backend._search_documents_from_prompt")
    def test_paper_compare_ops_computes_integer_rounded_percentage_from_author_year_queries(
        self,
        mock_search_prompt: object,
        mock_search_title: object,
        mock_fetch_pdf: object,
    ) -> None:
        def _search_side_effect(query: str, *args: object, **kwargs: object) -> list[dict[str, str]]:
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
    def test_video_transcript_ops_extracts_command_from_timestamp_window(self, mock_segments: object) -> None:
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
    def test_video_transcript_ops_extracts_response_after_question(self, mock_segments: object) -> None:
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
    def test_video_transcript_ops_counts_letter_occurrences_in_transcript_phrase(self, mock_segments: object) -> None:
        mock_segments.return_value = [
            {"start": 28.0, "end": 34.0, "text": '"RED EEL"'},
        ]

        answer, evidence, provenance = _solve_video_transcript_ops(
            'Thirty seconds into the first episode, a phrase is shown on the screen in white letters on a red background. How many times does the letter "E" appear in this phrase? https://www.youtube.com/watch?v=demo789'
        )

        self.assertEqual(answer, "3")
        self.assertTrue(any("transcript answer=3" in item for item in evidence))
        self.assertIn("youtube:transcript", provenance)

    @patch("domains.gaia_ops.backend._wayback_snapshot_html")
    @patch("domains.gaia_ops.backend._search_documents_from_prompt")
    def test_web_archive_ops_extracts_removed_menu_item(self, mock_search_docs: object, mock_snapshot_html: object) -> None:
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
        self, mock_search_docs: object, mock_snapshot_html: object
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
    def test_image_vision_ops_extracts_fraction_list(self, mock_lines: object) -> None:
        mock_lines.return_value = ["1/2 3/4", "noise"]

        answer, evidence, provenance = _solve_image_vision_ops(
            "List the fractions shown in the image.",
            [Path("fractions.png")],
        )

        self.assertEqual(answer, "1/2,3/4")
        self.assertTrue(any("fractions from fractions.png" in item for item in evidence))
        self.assertEqual(provenance, ["image:fractions.png"])

    @patch("domains.gaia_ops.backend._easyocr_text_lines")
    def test_image_vision_ops_extracts_latest_year(self, mock_lines: object) -> None:
        mock_lines.return_value = ["1894", "2003", "1998"]

        answer, evidence, provenance = _solve_image_vision_ops(
            "In the attached image, what is the latest chronological year that appears?",
            [Path("poster.jpg")],
        )

        self.assertEqual(answer, "2003")
        self.assertTrue(any("years from poster.jpg" in item for item in evidence))
        self.assertEqual(provenance, ["image:poster.jpg"])

    @patch("domains.gaia_ops.backend._solve_github_contributor_name_match")
    def test_github_public_artifact_ops_routes_contributor_match_structurally(self, mock_solver: object) -> None:
        mock_solver.return_value = ("Zhao Ziyang", ["matched contributor"])

        answer, evidence, provenance = _solve_github_public_artifact_ops(
            "Which contributor to the version of OpenCV where support was added for the Mask-RCNN model has the same name as a former Chinese head of government when the names are transliterated to the Latin alphabet?"
        )

        self.assertEqual(answer, "Zhao Ziyang")
        self.assertTrue(any("matched contributor" in item for item in evidence))
        self.assertEqual(provenance, ["github:contributors", "reference:public-entity-match"])

    @patch("domains.gaia_ops.backend._solve_generic_public_reference")
    def test_solve_question_generic_public_reference_requires_stronger_confidence_before_candidate_answer(
        self, mock_solve: object
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

        self.assertTrue(result["ok"])
        self.assertNotIn("solved", result)
        self.assertEqual(result["payload"]["candidate_answer"], "")
        self.assertEqual(result["payload"]["state_metadata"]["answer_mode"], "generic_public_reference")
        self.assertLess(result["payload"]["state_metadata"]["answer_confidence"], 0.72)

    @patch("domains.gaia_ops.backend._solve_generic_public_reference")
    def test_solve_question_generic_public_reference_exposes_candidate_answer_when_evidence_is_strong(
        self, mock_solve: object
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

        action = backend.fallback_repairs(state)[0]

        self.assertEqual(action.type, ActionType.BACKTRACK)
