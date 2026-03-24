from __future__ import annotations

import unittest
from types import SimpleNamespace

from engine.prompting import build_search_prompt
from engine.state import ReasoningState


class PromptCompactionTests(unittest.TestCase):
    def test_compact_prompt_uses_working_memory_view(self) -> None:
        state = ReasoningState(
            task_id="task_prompt_compact",
            domain="gaia_json_reasoning",
            problem_text=(
                "I am researching a long benchmark question with narrative filler. "
                "Return the answer as a comma-separated list in alphabetical order after checking the relevant evidence source. "
                "Use the latest 2022 English Wikipedia if necessary. "
                "There is a long middle section that should not dominate the whole prompt. " * 6
                + "\nWorkspace files:\n- report.xlsx\n- notes.txt\n- image.png"
            ),
            goal="Return the shortest correct answer",
            derived_facts=[
                "fact one about the target entity",
                "fact two about the supporting source",
                "fact three with a candidate answer",
                "fact four with a date filter",
                "fact five should force truncation",
            ],
            subgoals=["find the relevant page", "extract the date-bounded count"],
            obligations=["inspect the relevant source", "format the final answer"],
            evidence_refs=["page:alpha", "page:beta", "page:gamma", "page:delta", "page:epsilon"],
            tool_history=[
                {"tool": "search_web", "result": {"result": "found candidate page", "goal_progress": 0.2}},
                {"tool": "inspect_file", "result": {"answer": "report.xlsx contains filtered table", "goal_progress": 0.5}},
                {"tool": "solve_question", "result": {"payload": {"candidate_answer": "alpha, beta"}, "goal_progress": 0.7}},
            ],
            action_history=[
                {"type": "APPLY", "tool": "search_web", "content": "task prompt"},
                {"type": "APPLY", "tool": "inspect_file", "content": "report.xlsx"},
                {"type": "ANSWER", "content": "alpha, beta"},
            ],
            metadata={
                "target_file": "report.xlsx",
                "candidate_files": ["report.xlsx", "notes.txt", "image.png"],
                "inspected_files": ["report.xlsx"],
                "question_intent": "count and filter",
                "candidate_answer": "alpha, beta",
                "answer_confidence": 0.73,
                "reasoning_schema": {
                    "source_family": "public_reference",
                    "operator": "count with date filter",
                    "time_anchor": "latest 2022 version",
                    "output_contract": "comma-separated list with no whitespace",
                },
                "augmentation_layer": {
                    "mode": "trillion_structural",
                    "mindset": "recurse on hidden structure before accepting the first plausible answer",
                    "recursion": "time -> source -> operator -> rival -> contract",
                    "motif": "public_reference::count",
                    "source_order": "page -> section/table -> rival answer check",
                    "synthesis": "answer only when source/time/operator agree",
                    "output_guard": "comma-separated list with no whitespace",
                },
                "task_algebra": {
                    "equation": "time x source x operator x contract x rival",
                    "time_axis": "snapshot_year:2022",
                    "source_axis": "public_reference",
                    "operator_axis": "count",
                    "contract_axis": "comma-separated list with no whitespace",
                    "operator_stack": "retrieve pages -> isolate section/table -> apply operator -> contract check",
                    "closure_rule": "answer only when one candidate survives the rival check and the output contract",
                },
                "internal_role_machine": {
                    "roles": "framer -> retriever -> resolver -> judge -> closer",
                    "framer": "lock time/source/operator/contract",
                    "retriever": "collect only relevant evidence",
                    "resolver": "generate a small rival set",
                    "judge": "reject candidates that fail alignment",
                    "closer": "release only the surviving candidate",
                },
                "answer_self_check": {
                    "accepted": True,
                    "support": 0.66,
                    "notes": [
                        "evidence trail present",
                        "provenance present",
                        "contract=comma-separated list with no whitespace",
                    ],
                },
                "prompt_compaction": {
                    "enabled": True,
                    "problem_chars": 420,
                    "fact_limit": 3,
                    "subgoal_limit": 2,
                    "obligation_limit": 2,
                    "evidence_limit": 3,
                    "tool_limit": 2,
                    "action_limit": 2,
                    "file_limit": 3,
                    "retrieval_item_limit": 1,
                    "text_item_chars": 70,
                },
            },
        )

        prompt = build_search_prompt(
            state,
            'ACTION {"type":"ANSWER","content":"final answer"}',
            retrieval_context={
                "lemmas": [SimpleNamespace(name="lemma_a", pattern="important pattern", tactic_chain=["inspect", "count"])],
                "hard_cases": [{"task": "very long hard case description " * 8, "answer": "wrong", "expected": "right"}],
                "tool_priors": {"inspect_file": 0.8, "search_web": 0.7},
                "failure_avoidance": ["do not answer before checking the relevant source"],
            },
            tactic_hints=["inspect_file bias=0.91", "solve_question bias=0.82"],
        )

        self.assertIn("[TASK]", prompt)
        self.assertIn("[FOCUS]", prompt)
        self.assertIn("[RECENT_TOOLS]", prompt)
        self.assertIn("[TACTIC_HINTS]", prompt)
        self.assertIn("[AUGMENTATION]", prompt)
        self.assertIn("[TASK_ALGEBRA]", prompt)
        self.assertIn("[ROLE_MACHINE]", prompt)
        self.assertIn("[REASONING_SCHEMA]", prompt)
        self.assertIn("[SELF_CHECK]", prompt)
        self.assertNotIn("[METADATA]", prompt)
        self.assertLess(len(prompt), len(state.serialize()) + 200)
        self.assertIn("target_file=report.xlsx", prompt)
        self.assertIn("candidate_answer=alpha, beta", prompt)
        self.assertIn("mode=trillion_structural", prompt)
        self.assertIn("mindset=recurse on hidden structure", prompt)
        self.assertIn("equation=time x source x operator x contract x rival", prompt)
        self.assertIn("roles=framer -> retriever -> resolver -> judge -> closer", prompt)
        self.assertIn("source_family=public_reference", prompt)
        self.assertIn("contract=comma-separated list with no whitespace", prompt)
        self.assertNotIn("fact one about the target entity | fact two about the supporting source | fact three with a candidate answer | fact four with a date filter | fact five should force truncation", prompt)

    def test_prompt_defaults_to_full_state_without_compaction(self) -> None:
        state = ReasoningState(
            task_id="task_prompt_full",
            domain="math",
            problem_text="Compute 2 + 2",
            goal="Return the answer",
            metadata={},
        )

        prompt = build_search_prompt(state, 'ACTION {"type":"ANSWER","content":"4"}')

        self.assertTrue(prompt.startswith(state.serialize()))


if __name__ == "__main__":
    unittest.main()
