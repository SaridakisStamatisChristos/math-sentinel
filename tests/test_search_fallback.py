from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from proof.actions import ActionType
from proof.executor import ProofExecutor
from proof.state import ProofState
from search.beam import beam_search
from tools.registry import ToolRegistry


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


class BeamFallbackTests(unittest.TestCase):
    def test_invalid_generation_uses_answer_repair_for_arithmetic(self) -> None:
        initial_state = ProofState(
            task_id="arith_1",
            domain="arithmetic",
            problem_text="Compute: 2 + 3",
            goal="Compute the integer result",
            expected_answer="5",
            metadata={"family": "arithmetic"},
        )

        with patch("search.beam.propose_actions", return_value=["not an action"]):
            final_state, explored = beam_search(
                prover=object(),
                verifier=DummyVerifier(),
                tokenizer=DummyTokenizer(),
                executor=ProofExecutor(ToolRegistry()),
                initial_state=initial_state,
                device="cpu",
                beam_width=2,
                max_depth=1,
                proposal_count=1,
            )

        self.assertEqual(final_state.status, "solved")
        self.assertEqual(final_state.final_answer, "5")
        self.assertEqual(len(explored), 2)
        self.assertIsNotNone(explored[1].action)
        self.assertEqual(explored[1].action.type, ActionType.ANSWER)

    def test_invalid_generation_uses_terminal_tool_repair_for_derivative(self) -> None:
        initial_state = ProofState(
            task_id="der_1",
            domain="derivative",
            problem_text="Differentiate with respect to x: x**2 + 3*x",
            goal="Return d/dx",
            expected_answer="2*x + 3",
            metadata={"family": "derivative"},
        )

        with patch("search.beam.propose_actions", return_value=["not an action"]):
            final_state, explored = beam_search(
                prover=object(),
                verifier=DummyVerifier(),
                tokenizer=DummyTokenizer(),
                executor=ProofExecutor(ToolRegistry()),
                initial_state=initial_state,
                device="cpu",
                beam_width=2,
                max_depth=2,
                proposal_count=1,
            )

        self.assertEqual(final_state.status, "solved")
        self.assertEqual(final_state.final_answer.replace(" ", ""), "2*x+3")
        self.assertEqual(explored[1].action.type, ActionType.APPLY)


if __name__ == "__main__":
    unittest.main()