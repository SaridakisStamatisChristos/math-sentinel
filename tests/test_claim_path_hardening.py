from __future__ import annotations

import unittest

from benchmarks.runners import _claim_profile_requirements
from domains import create_reasoning_domain
from memory.hard_cases import HardCaseStore
from memory.replay import ReplayBuffer
from memory.retrieval import retrieve_context
from sentinel.config import load_runtime_config
from train_v7 import benchmark_tasks_for_backend, sample_benchmark_recovery_examples


class _LemmaStore:
    def retrieve(self, domain: str, text: str, limit: int = 5):  # noqa: ARG002
        return []


class ClaimPathHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strict_cfg = load_runtime_config("config/benchmarks/profile_public_unassisted_strict.yaml", search_config_path="")

    def test_swebench_answer_candidates_do_not_expose_expected_answer(self) -> None:
        domain = create_reasoning_domain("swebench_ops", runtime_config=self.strict_cfg)
        task = domain.benchmark_tasks()[0]
        state = domain.make_state(task)

        answers = [binding["content"] for binding in domain.candidate_bindings(state, "ANSWER")]

        self.assertNotIn(task.answer, answers)

    def test_gaia_answer_candidates_do_not_expose_expected_answer(self) -> None:
        domain = create_reasoning_domain("gaia_ops", runtime_config=self.strict_cfg)
        task = domain.benchmark_tasks()[0]
        state = domain.make_state(task)

        answers = [binding["content"] for binding in domain.candidate_bindings(state, "ANSWER")]

        self.assertNotIn(task.answer, answers)

    def test_holdout_enabled_sampling_uses_private_train_cases(self) -> None:
        for backend_name in ["swebench_ops", "gaia_ops"]:
            domain = create_reasoning_domain(backend_name, runtime_config=self.strict_cfg)
            train_ids = {task.task_id for task in domain.training_tasks()}
            benchmark_ids = {task.task_id for task in domain.benchmark_tasks()}

            self.assertTrue(train_ids)
            self.assertTrue(benchmark_ids)
            self.assertTrue(train_ids.isdisjoint(benchmark_ids))

            sampled_ids = {domain.sample_task([]).task_id for _ in range(16)}

            self.assertTrue(sampled_ids)
            self.assertTrue(sampled_ids.issubset(train_ids))
            self.assertTrue(sampled_ids.isdisjoint(benchmark_ids))

    def test_train_case_harvest_target_uses_private_cases(self) -> None:
        domain = create_reasoning_domain("swebench_ops", runtime_config=self.strict_cfg)
        train_ids = {task.task_id for _, task in benchmark_tasks_for_backend(domain, ["train_cases"])}
        benchmark_ids = {task.task_id for task in domain.benchmark_tasks()}

        self.assertTrue(train_ids)
        self.assertTrue(train_ids.isdisjoint(benchmark_ids))

    def test_recovery_examples_ignore_public_holdout_failures(self) -> None:
        domain = create_reasoning_domain("swebench_ops", runtime_config=self.strict_cfg)
        replay = ReplayBuffer(capacity=8)
        public_task = domain.benchmark_tasks()[0]
        train_task = domain.training_tasks()[0]
        replay.add(
            {
                "kind": "benchmark_failure",
                "task_id": public_task.task_id,
                "domain": public_task.domain,
                "task": public_task.prompt,
                "goal": public_task.goal,
                "expected": public_task.answer,
                "meta": dict(public_task.meta),
                "source": "benchmark_claim_holdout",
                "holdout_group": public_task.meta.get("holdout_group", ""),
                "weight": 5.0,
            }
        )
        replay.add(
            {
                "kind": "benchmark_failure",
                "task_id": train_task.task_id,
                "domain": train_task.domain,
                "task": train_task.prompt,
                "goal": train_task.goal,
                "expected": train_task.answer,
                "meta": dict(train_task.meta),
                "source": "benchmark_train",
                "holdout_group": train_task.meta.get("holdout_group", ""),
                "weight": 5.0,
            }
        )

        examples = sample_benchmark_recovery_examples(replay, domain, limit=4)

        self.assertEqual(len(examples), 1)
        self.assertIn(train_task.prompt, examples[0])
        self.assertNotIn(public_task.prompt, examples[0])

    def test_claim_retrieval_filters_exclude_holdout_sources(self) -> None:
        store = HardCaseStore(capacity=8)
        store.add({"task": "public holdout issue", "domain": "swebench_patch", "source": "benchmark_claim_holdout", "suite": "swebench_verified_smoke"})
        store.add({"task": "private train issue", "domain": "swebench_patch", "source": "benchmark_train", "suite": "swebench_private_train"})

        context = retrieve_context(
            _LemmaStore(),
            store,
            "swebench_patch",
            "repair the repository",
            filters={"exclude_sources": ["benchmark_claim_holdout"]},
        )

        hard_case_tasks = [str(item.get("task", "")) for item in context["hard_cases"]]

        self.assertIn("private train issue", hard_case_tasks)
        self.assertNotIn("public holdout issue", hard_case_tasks)

    def test_claim_profile_requires_no_repair_or_chain(self) -> None:
        ok, failures = _claim_profile_requirements(
            self.strict_cfg,
            {
                "benchmark_integrity_passed": True,
                "fallback_chain_used": False,
                "fallback_repair_used": True,
                "guided_rollout_used": False,
            },
        )

        self.assertFalse(ok)
        self.assertIn("fallback_repair_used=true", failures)


if __name__ == "__main__":
    unittest.main()
