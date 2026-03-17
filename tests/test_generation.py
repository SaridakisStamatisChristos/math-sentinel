from __future__ import annotations

import unittest

import torch

from engine.state import ReasoningState
from sentinel.generation import build_structured_action_candidates, propose_structured_actions
from sentinel.tokenizer import build_default_tokenizer


class _ZeroModel:
    def __init__(self, seq_len: int, vocab_size: int) -> None:
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        batch, timesteps = x.shape
        return torch.zeros((batch, timesteps, self.vocab_size), dtype=torch.float32, device=x.device)


class StructuredGenerationTests(unittest.TestCase):
    def test_build_structured_candidates_include_grounded_apply_and_answer(self) -> None:
        state = ReasoningState(
            task_id="toy_1",
            domain="assignment_count",
            problem_text="Count assignments in the function",
            goal="Return the assignment count",
            metadata={"family": "assignment_count"},
        )
        state.derived_facts.append("2")

        candidates = build_structured_action_candidates(state, available_tools=["assignment_count", "function_name"])

        self.assertTrue(any('"tool":"assignment_count"' in candidate for candidate in candidates))
        self.assertTrue(any('"type":"ANSWER"' in candidate and '"content":"2"' in candidate for candidate in candidates))

    def test_propose_structured_actions_returns_canonical_lines(self) -> None:
        tokenizer = build_default_tokenizer()
        model = _ZeroModel(seq_len=192, vocab_size=tokenizer.vocab_size)
        state = ReasoningState(
            task_id="toy_2",
            domain="project_plan",
            problem_text="Create a valid project plan",
            goal="Return an ordered plan",
            metadata={"family": "project_plan"},
        )

        proposals = propose_structured_actions(
            model=model,
            tokenizer=tokenizer,
            prompt=state.serialize(),
            device="cpu",
            state=state,
            available_tools=["project_plan", "shopping_plan"],
            proposal_count=3,
            temperature=0.0,
        )

        self.assertGreaterEqual(len(proposals), 1)
        self.assertTrue(all(proposal.startswith("ACTION ") for proposal in proposals))


if __name__ == "__main__":
    unittest.main()
