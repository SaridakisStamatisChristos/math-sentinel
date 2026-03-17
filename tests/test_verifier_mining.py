from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from curriculum.generators import GeneratedTask
from memory.replay import ReplayBuffer
from proof.state import ProofState
from search.nodes import SearchNode
from train_v7 import build_verifier_targets, mine_online_verifier_pairs, pick_best_mined_pair


def _task() -> GeneratedTask:
    return GeneratedTask(
        task_id="t1",
        domain="linear_equation",
        prompt="Solve: 2x + 3 = 11",
        answer="x=4",
        goal="Solve for x",
        meta={"family": "linear_equation"},
    )


def test_build_verifier_targets_rewards_correct_solution() -> None:
    task = _task()
    solved = ProofState(task_id=task.task_id, domain=task.domain, problem_text=task.prompt, goal=task.goal, expected_answer=task.answer, metadata=task.meta)
    solved.final_answer = "x=4"
    solved.status = "solved"
    solved.derived_facts.append("x=4")

    open_state = ProofState(task_id=task.task_id, domain=task.domain, problem_text=task.prompt, goal=task.goal, expected_answer=task.answer, metadata=task.meta)

    solved_targets = build_verifier_targets(task, solved)
    open_targets = build_verifier_targets(task, open_state, local_scores={"valid_step": 0.35, "risk_score": 0.8})

    assert solved_targets[2].item() > open_targets[2].item()
    assert solved_targets[3].item() < open_targets[3].item()


def test_pick_best_mined_pair_uses_wrong_answer_node_as_negative() -> None:
    task = _task()

    good_state = ProofState(task_id=task.task_id, domain=task.domain, problem_text=task.prompt, goal=task.goal, expected_answer=task.answer, metadata=task.meta)
    good_state.final_answer = "x=4"
    good_state.status = "solved"
    good_state.derived_facts.append("x=4")

    bad_state = ProofState(task_id=task.task_id, domain=task.domain, problem_text=task.prompt, goal=task.goal, expected_answer=task.answer, metadata=task.meta)
    bad_state.final_answer = "x=5"
    bad_state.status = "solved"
    bad_state.derived_facts.extend(["2x=8", "candidate=x=5"])

    explored = [
        SearchNode(state=good_state, cumulative_score=2.0, local_scores={"valid_step": 1.0, "goal_progress": 1.0}, depth=1),
        SearchNode(state=bad_state, cumulative_score=1.5, local_scores={"valid_step": 0.9, "goal_progress": 0.8}, depth=1),
    ]

    pair = pick_best_mined_pair(task, explored, good_state)

    assert pair is not None
    assert "x=4" in pair["pos_text"]
    assert "x=5" in pair["neg_text"]


def test_pick_best_mined_pair_falls_back_to_root_state() -> None:
    task = _task()

    root = ProofState(task_id=task.task_id, domain=task.domain, problem_text=task.prompt, goal=task.goal, expected_answer=task.answer, metadata=task.meta)
    solved = ProofState(task_id=task.task_id, domain=task.domain, problem_text=task.prompt, goal=task.goal, expected_answer=task.answer, metadata=task.meta)
    solved.final_answer = "x=4"
    solved.status = "solved"
    solved.derived_facts.append("x=4")

    pair = pick_best_mined_pair(
        task,
        [
            SearchNode(state=root, cumulative_score=0.0, local_scores={}, depth=0),
            SearchNode(state=solved, cumulative_score=2.0, local_scores={"valid_step": 1.0, "goal_progress": 1.0}, depth=1),
        ],
        solved,
    )

    assert pair is not None
    assert "[FINAL_ANSWER] none" in pair["neg_text"]


def test_online_verifier_mining_uses_passed_search_config() -> None:
    task = _task()
    root = ProofState(task_id=task.task_id, domain=task.domain, problem_text=task.prompt, goal=task.goal, expected_answer=task.answer, metadata=task.meta)
    solved = ProofState(task_id=task.task_id, domain=task.domain, problem_text=task.prompt, goal=task.goal, expected_answer=task.answer, metadata=task.meta)
    solved.final_answer = "x=4"
    solved.status = "solved"
    solved.derived_facts.append("x=4")
    explored = [
        SearchNode(state=root, cumulative_score=0.0, local_scores={}, depth=0),
        SearchNode(state=solved, cumulative_score=2.0, local_scores={"valid_step": 1.0, "goal_progress": 1.0}, depth=1),
    ]

    replay = ReplayBuffer(capacity=4)
    prover = torch.nn.Linear(2, 2)
    verifier = torch.nn.Linear(2, 2)
    phase = SimpleNamespace(domains=["linear_equation"])

    with patch("train_v7.sample_task", return_value=task):
        with patch("train_v7.beam_search", return_value=(solved, explored)) as mock_beam_search:
            mined = mine_online_verifier_pairs(
                prover=prover,
                verifier=verifier,
                tokenizer=object(),
                executor=object(),
                phase=phase,
                replay=replay,
                device="cpu",
                count=1,
                beam_width=2,
                max_depth=2,
                proposal_count=2,
                max_new_tokens=16,
                temperature=0.8,
                top_k=8,
                score_config={"solved_bonus": 9.0},
            )

    assert len(mined) == 1
    assert len(replay.items) == 1
    assert mock_beam_search.call_args.kwargs["score_config"] == {"solved_bonus": 9.0}
