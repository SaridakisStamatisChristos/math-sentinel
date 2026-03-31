from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from domains.gaia_ops.query_runtime import GaiaOperator, GaiaQueryEngine, GaiaSolveContext, get_active_gaia_context


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"


class GaiaQueryRuntimeTests(unittest.TestCase):
    def _fresh_dir(self, name: str) -> Path:
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        path = TEST_TMP_ROOT / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def test_run_stage_writes_progress_and_resume_snapshot(self) -> None:
        root = self._fresh_dir("gaia-query-runtime")
        progress_path = root / "progress.jsonl"
        resume_path = root / "resume.json"

        def _handler(arg: str, state: object | None = None) -> dict[str, object]:
            return {"ok": True, "result": str(arg)}

        engine = GaiaQueryEngine(
            {
                "plan_question": GaiaOperator(
                    name="plan_question",
                    handler=_handler,
                    phase="plan",
                    description="plan",
                )
            }
        )
        context = GaiaSolveContext(
            task_id="runtime_case",
            prompt="trace the source and answer with a title",
            workspace_dir=str(root),
            available_files=["TASK.md"],
            metadata={"target_file": "", "candidate_files": []},
            question_plan={"research_mode": "generic_public_reference"},
            progress_log_path=str(progress_path),
            resume_snapshot_path=str(resume_path),
            operator_names=["plan_question"],
            resume_enabled=True,
        )

        result = engine.run_stage(
            "plan",
            context,
            lambda ctx: {
                "ok": True,
                "result": "locate the public page then answer",
                "payload": {
                    "state_metadata": {
                        "question_plan": {"research_mode": "generic_public_reference"},
                        "answer_confidence": 0.66,
                    }
                },
            },
            compact_state={
                "task_id": "runtime_case",
                "question": "trace the source and answer with a title",
                "research_mode": "generic_public_reference",
                "answer_contract": "title",
            },
        )

        self.assertTrue(progress_path.exists())
        self.assertTrue(resume_path.exists())
        self.assertIn("gaia_compact_state", result["payload"]["state_metadata"])
        self.assertEqual(
            result["payload"]["state_metadata"]["gaia_compact_state"]["research_mode"],
            "generic_public_reference",
        )
        self.assertEqual(
            result["payload"]["state_metadata"]["gaia_runtime_stage"],
            "plan",
        )
        self.assertTrue(result["payload"]["state_metadata"]["gaia_recent_progress"])
        progress_lines = progress_path.read_text(encoding="utf-8").splitlines()
        self.assertGreaterEqual(len(progress_lines), 2)
        snapshot = json.loads(resume_path.read_text(encoding="utf-8"))
        self.assertEqual(snapshot["task_id"], "runtime_case")
        self.assertEqual(snapshot["stage"], "plan")
        self.assertEqual(snapshot["compact_state"]["answer_contract"], "title")

    def test_active_context_is_available_inside_stage(self) -> None:
        root = self._fresh_dir("gaia-query-runtime-active")

        def _handler(arg: str, state: object | None = None) -> dict[str, object]:
            return {"ok": True, "result": str(arg)}

        engine = GaiaQueryEngine(
            {
                "solve_question": GaiaOperator(
                    name="solve_question",
                    handler=_handler,
                    phase="solve",
                    description="solve",
                )
            }
        )
        context = GaiaSolveContext(
            task_id="active_case",
            prompt="solve",
            workspace_dir=str(root),
            available_files=[],
            metadata={},
        )

        def _callback(ctx: GaiaSolveContext) -> dict[str, object]:
            active = get_active_gaia_context()
            self.assertIs(active, ctx)
            active.emit("manual_probe", status="ok")
            return {"ok": True, "result": "done", "payload": {"state_metadata": {}}}

        result = engine.run_stage("solve", context, _callback, compact_state={"task_id": "active_case", "question": "solve"})

        self.assertTrue(result["payload"]["state_metadata"]["gaia_recent_progress"])


if __name__ == "__main__":
    unittest.main()
