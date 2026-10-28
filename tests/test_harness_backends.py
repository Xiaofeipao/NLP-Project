from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from ppnl.core import Grid
from ppnl.harness_backends import PolicyBackend, RepairPolicyBackend


class BackendTests(unittest.TestCase):
    def test_recovery_switches_after_persisted_failed_calls(self):
        repair, policy = Mock(), Mock()
        backend = RepairPolicyBackend(repair, policy, 2)
        context = {"state": {"tool_calls": 1, "plan": ""}}
        self.assertIs(backend.next_call(context), repair.next_call.return_value)
        context["state"]["tool_calls"] = 2
        self.assertIs(backend.next_call(context), policy.next_call.return_value)
        self.assertEqual(repair.next_call.call_count, 1)
        self.assertEqual(policy.next_call.call_count, 1)

    def test_committed_recovery_actions_cannot_restart_whole_plan(self):
        repair, policy = Mock(), Mock()
        backend = RepairPolicyBackend(repair, policy, 2)
        backend.next_call({"state": {"tool_calls": 0, "plan": "right"}})
        repair.next_call.assert_not_called()
        policy.next_call.assert_called_once()

    def test_revisit_mask_prefers_unvisited_legal_neighbor(self):
        grid = Grid(2, 3, (), (0, 0), (1, 2))
        scorer = SimpleNamespace(scores=lambda observations: [[0, 1, 100, 2]])
        backend = PolicyBackend(scorer, True, True)
        result = backend.next_call({"grid": asdict(grid), "position": [0, 1],
                                    "state": {"plan": "right"}})
        self.assertEqual(result["arguments"]["action"], "right")

    def test_revisit_mask_allows_backtracking_from_dead_end(self):
        grid = Grid(2, 3, ((1, 0), (1, 1)), (0, 0), (1, 2))
        scorer = SimpleNamespace(scores=lambda observations: [[0, 0, 0, 10]])
        backend = PolicyBackend(scorer, True, True)
        result = backend.next_call({"grid": asdict(grid), "position": [0, 0],
                                    "state": {"plan": "right left"}})
        self.assertEqual(result["arguments"]["action"], "right")

    def test_recovery_threshold_and_mask_are_validated(self):
        for value in (0, True, 1.2):
            with self.assertRaises(ValueError):
                RepairPolicyBackend(Mock(), Mock(), value)
        with self.assertRaises(ValueError):
            PolicyBackend(Mock(), False, True)
