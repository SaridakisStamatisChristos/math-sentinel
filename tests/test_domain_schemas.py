from __future__ import annotations

import unittest

from domains.code_ops.backend import CodeOpsReasoningDomain
from domains.math.backend import MathReasoningDomain
from domains.planning_ops.backend import PlanningOpsReasoningDomain
from domains.string_ops.backend import StringOpsReasoningDomain


class DomainSchemaTests(unittest.TestCase):
    def test_backends_expose_non_empty_action_schemas(self) -> None:
        backends = [
            MathReasoningDomain(),
            StringOpsReasoningDomain(),
            CodeOpsReasoningDomain(),
            PlanningOpsReasoningDomain(),
        ]

        for backend in backends:
            task = backend.benchmark_tasks()[0]
            state = backend.make_state(task)
            schema = backend.action_schema(state)
            self.assertTrue(schema["strict"])
            self.assertGreaterEqual(len(backend.allowed_action_types(state)), 1)


if __name__ == "__main__":
    unittest.main()
