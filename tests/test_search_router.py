from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from domains.planning_ops.backend import PlanningOpsReasoningDomain
from proof.actions import Action, ActionType
from proof.state import ProofState
from search.mcts import mcts_search
from search.router import resolve_search_mode, run_search


class DummyTokenizer:
    def encode(self, text: str, seq_len: int) -> list[int]:
        return [0] * seq_len


class DummyVerifier:
    def predict_scores(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = x.shape[0]
        return {
            "valid_step_prob": torch.full((batch,), 0.8),
            "goal_progress_score": torch.full((batch,), 0.6),
            "proof_completion_score": torch.full((batch,), 0.4),
            "risk_score": torch.full((batch,), 0.1),
            "branch_priority": torch.full((batch,), 0.7),
        }


class SearchRouterTests(unittest.TestCase):
    def test_auto_mode_prefers_mcts_for_planning_domains(self) -> None:
        planning_state = ProofState(
            task_id="plan_1",
            domain="project_plan",
            problem_text="Create a valid project plan",
            goal="Return an ordered plan",
            metadata={"family": "project_plan"},
        )
        arithmetic_state = ProofState(
            task_id="arith_1",
            domain="arithmetic",
            problem_text="Compute: 2 + 3",
            goal="Compute the integer result",
            metadata={"family": "arithmetic"},
        )

        self.assertEqual(resolve_search_mode(planning_state, {"mode": "auto"}), "mcts")
        self.assertEqual(resolve_search_mode(arithmetic_state, {"mode": "auto"}), "beam")

    def test_run_search_routes_to_expected_engine(self) -> None:
        state = ProofState(
            task_id="plan_2",
            domain="project_plan",
            problem_text="Create a valid project plan",
            goal="Return an ordered plan",
            metadata={"family": "project_plan"},
        )

        with patch("search.router.mcts_search", return_value=("mcts", [])) as mock_mcts:
            result = run_search(initial_state=state, score_config={"mode": "auto"})

        self.assertEqual(result, ("mcts", []))
        self.assertTrue(mock_mcts.called)

    def test_mcts_uses_fallback_repairs_when_generation_is_invalid(self) -> None:
        backend = PlanningOpsReasoningDomain()
        state = ProofState(
            task_id="plan_3",
            domain="project_plan",
            problem_text="Create a valid project plan.\nTasks:\n- design (duration=1, priority=3, deps=none)\n- build (duration=2, priority=4, deps=design)\n- test (duration=1, priority=2, deps=build)\nReturn the ordered task plan.",
            goal="Return the ordered task plan",
            expected_answer="design -> build -> test",
            metadata={"family": "project_plan"},
        )

        def repair_fn(_: ProofState) -> list[Action]:
            return [Action(type=ActionType.APPLY, tool="project_plan", content=state.problem_text)]

        with patch("search.mcts.propose_actions", return_value=["not an action"]):
            final_state, explored = mcts_search(
                prover=object(),
                verifier=DummyVerifier(),
                tokenizer=DummyTokenizer(),
                executor=backend.create_executor(),
                initial_state=state,
                device="cpu",
                max_depth=2,
                proposal_count=1,
                score_config={"decoder_mode": "free", "mcts_simulations": 4, "mcts_branching": 2},
                fallback_repairs_fn=repair_fn,
            )

        self.assertEqual(final_state.status, "solved")
        self.assertEqual(final_state.final_answer, "design -> build -> test")
        self.assertGreaterEqual(len(explored), 2)


if __name__ == "__main__":
    unittest.main()
