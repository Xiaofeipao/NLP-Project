"""Runtime contract tests, using scripted tools only; no models or GPU required."""
import json
from pathlib import Path
import tempfile
import unittest

from ppnl.core import Grid
from ppnl.harness import Budgets, create_session, read_session, run_session


class Scripted:
    def __init__(self, calls):
        self.calls = iter(calls)

    def next_call(self, context):
        call = next(self.calls)
        if isinstance(call, Exception):
            raise call
        return call


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


class AgentHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "trial"
        self.grid = Grid(2, 2, (), (0, 0), (1, 1))
        self.identity = {"kind": "scripted_fixture"}

    def create(self, **budgets):
        create_session(self.directory, self.grid, Budgets(**budgets), self.identity, {"kind": "local"})

    def run_calls(self, *calls):
        return run_session(self.directory, Scripted(calls), self.identity)

    def test_moves_are_verified_and_stop_at_goal(self):
        self.create()
        state = self.run_calls(call("move", action="right"), call("move", action="down"))
        self.assertEqual(state["status"], "success")
        self.assertEqual(state["plan"], "right down")
        self.assertEqual(state["action_attempts"], 2)

    def test_collisions_do_not_change_committed_position(self):
        self.create()
        state = self.run_calls(call("move", action="up"), call("submit_plan", plan="down right"))
        self.assertEqual(state["plan"], "down right")
        self.assertEqual(state["invalid_calls"], 1)

    def test_agent_claim_cannot_create_success(self):
        self.create()
        state = self.run_calls(call("finish"))
        self.assertEqual(state["status"], "stopped")
        self.assertEqual(state["plan"], "")

    def test_verification_does_not_commit_a_proposal(self):
        self.create()
        state = self.run_calls(call("verify_plan", plan="down right"), call("finish"))
        self.assertEqual(state["plan"], "")
        self.assertEqual(state["status"], "stopped")

    def test_model_budget_blocks_another_call(self):
        self.create(model_calls=1)
        state = self.run_calls(call("move", action="right"))
        self.assertEqual(state["status"], "budget_exhausted")
        self.assertEqual(state["model_calls"], 1)

    def test_action_budget_rejects_entire_plan(self):
        self.create(action_attempts=1)
        state = self.run_calls(call("submit_plan", plan="down right"))
        self.assertEqual(state["status"], "budget_exhausted")
        self.assertEqual(state["plan"], "")

    def test_unknown_tool_and_extra_arguments_are_rejected(self):
        self.create(tool_calls=2)
        state = self.run_calls(call("shell", command="anything"), call("move", action="down", bypass=True))
        self.assertEqual(state["status"], "budget_exhausted")
        self.assertEqual(state["invalid_calls"], 2)

    def test_resume_uses_journal_even_with_stale_snapshot(self):
        self.create()
        with self.assertRaises(RuntimeError):
            self.run_calls(call("move", action="right"), RuntimeError("interrupted"))
        (self.directory / "state.json").write_text("{}", encoding="utf-8")
        state = self.run_calls(call("move", action="down"))
        self.assertEqual(state["status"], "success")
        self.assertEqual(state["plan"], "right down")
        self.assertEqual(state["model_calls"], 3)

    def test_resume_refuses_changed_backend(self):
        self.create()
        with self.assertRaises(ValueError):
            run_session(self.directory, Scripted([]), {"kind": "changed"})

    def test_torn_or_modified_journal_is_not_silently_replayed(self):
        self.create()
        self.run_calls(call("finish"))
        journal = self.directory / "events.jsonl"
        journal.write_bytes(journal.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            read_session(self.directory)

    def test_manifest_tampering_is_detected(self):
        self.create()
        path = self.directory / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["budgets"]["model_calls"] = 999
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            read_session(self.directory)

    def test_existing_trial_is_never_overwritten(self):
        self.create()
        with self.assertRaises(FileExistsError):
            self.create()

    def test_budgets_reject_boolean_fraction_nan_and_zero(self):
        for kwargs in ({"tool_calls": True}, {"model_calls": 1.5},
                       {"active_seconds": float("nan")}, {"context_events": 0}):
            with self.assertRaises(ValueError):
                Budgets(**kwargs)


if __name__ == "__main__":
    unittest.main()
