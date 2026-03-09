from __future__ import annotations

import unittest

from proof.actions import ActionType
from proof.parser import parse_actions


class ParseActionsTests(unittest.TestCase):
    def test_parses_multiple_valid_actions(self) -> None:
        text = (
            '<action type="THINK">Plan the step.</action>'
            '<action type="APPLY" tool="add">2,3</action>'
            '<action type="ANSWER">5</action>'
        )

        actions, confidence = parse_actions(text)

        self.assertEqual([action.type for action in actions], [ActionType.THINK, ActionType.APPLY, ActionType.ANSWER])
        self.assertEqual(actions[1].tool, "add")
        self.assertEqual(actions[2].content, "5")
        self.assertEqual(confidence, 1.0)

    def test_parses_answer_tag_fallback(self) -> None:
        actions, confidence = parse_actions("scratch work\n<answer>x=4</answer>")

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].type, ActionType.ANSWER)
        self.assertEqual(actions[0].content, "x=4")
        self.assertEqual(confidence, 0.55)

    def test_drops_invalid_apply_action_without_tool(self) -> None:
        actions, confidence = parse_actions('<action type="APPLY">2,3</action>')

        self.assertEqual(actions, [])
        self.assertEqual(confidence, 0.0)


if __name__ == "__main__":
    unittest.main()