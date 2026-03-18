from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import torch

from memory.hard_cases import HardCaseStore
from memory.lemma_store import LemmaStore
from memory.replay import ReplayBuffer
from memory.tactic_stats import TacticStats
from proof.lemmas import Lemma
from sentinel.checkpointing import load_checkpoint, load_checkpoint_metadata, save_checkpoint


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"


class ResumePersistenceTests(unittest.TestCase):
    def _fresh_dir(self, name: str) -> Path:
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        path = TEST_TMP_ROOT / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def test_memory_stores_round_trip_from_disk(self) -> None:
        root = self._fresh_dir("resume-memory")

        replay = ReplayBuffer(capacity=3)
        replay.add({"task": "a"})
        replay.add({"task": "b"})
        replay.save_jsonl(str(root / "replay.jsonl"))

        hard_cases = HardCaseStore(capacity=2)
        hard_cases.add({"domain": "arith", "score": 1.0})
        hard_cases.save(str(root / "hard.json"))

        lemma_store = LemmaStore()
        lemma_store.add(Lemma(name="lin", pattern="solve", tactic_chain=["rewrite", "answer"], domains=["linear_equation"], success_count=2))
        lemma_store.save(str(root / "lemma.json"))

        tactic_stats = TacticStats()
        tactic_stats.record("arith", "ANSWER", True)
        tactic_stats.save(str(root / "tactic.json"))

        replay_loaded = ReplayBuffer(capacity=3)
        replay_loaded.load_jsonl(str(root / "replay.jsonl"))
        hard_loaded = HardCaseStore(capacity=2)
        hard_loaded.load(str(root / "hard.json"))
        lemma_loaded = LemmaStore()
        lemma_loaded.load(str(root / "lemma.json"))
        tactic_loaded = TacticStats()
        tactic_loaded.load(str(root / "tactic.json"))

        self.assertEqual([item["task"] for item in replay_loaded.items], ["a", "b"])
        self.assertEqual(hard_loaded.cases[0]["domain"], "arith")
        self.assertIn("lin", lemma_loaded.lemmas)
        self.assertGreater(tactic_loaded.bias("arith", "ANSWER"), 0.5)

    def test_checkpoint_round_trip_restores_scaler_and_optimizers(self) -> None:
        root = self._fresh_dir("resume-checkpoint")
        path = str(root / "last.pt")
        prover = torch.nn.Linear(2, 2)
        verifier = torch.nn.Linear(2, 1)
        prover_optim = torch.optim.AdamW(prover.parameters(), lr=1e-3)
        verifier_optim = torch.optim.AdamW(verifier.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        save_checkpoint(
            path,
            prover=prover,
            verifier=verifier,
            prover_optim=prover_optim,
            verifier_optim=verifier_optim,
            scaler=scaler,
            step=7,
            config={"seed": 1},
            extra_state={"phase": "smoke"},
        )

        loaded_prover = torch.nn.Linear(2, 2)
        loaded_verifier = torch.nn.Linear(2, 1)
        loaded_prover_optim = torch.optim.AdamW(loaded_prover.parameters(), lr=1e-3)
        loaded_verifier_optim = torch.optim.AdamW(loaded_verifier.parameters(), lr=1e-3)
        loaded_scaler = torch.amp.GradScaler("cuda", enabled=False)

        payload = load_checkpoint(
            path,
            loaded_prover,
            loaded_verifier,
            loaded_prover_optim,
            loaded_verifier_optim,
            scaler=loaded_scaler,
        )

        self.assertEqual(payload["step"], 7)
        self.assertEqual(payload["extra_state"]["phase"], "smoke")
        self.assertEqual(loaded_scaler.state_dict(), scaler.state_dict())
        self.assertEqual(load_checkpoint_metadata(path)["provider"], "legacy_tiny")

    def test_legacy_checkpoint_metadata_defaults_when_missing(self) -> None:
        root = self._fresh_dir("resume-legacy-metadata")
        path = str(root / "legacy.pt")
        torch.save({"step": 1, "prover": {}, "verifier": {}}, path)

        metadata = load_checkpoint_metadata(path)

        self.assertEqual(metadata["provider"], "legacy_tiny")
        self.assertEqual(metadata["backbone"], "legacy_tiny")


if __name__ == "__main__":
    unittest.main()
