import unittest

from ppnl.core import Grid, execute
from state_baselines.prepare_step_verification import observations
from state_baselines.step_verification import step_prompt, verify_steps


class StepVerificationTests(unittest.TestCase):
    grid = Grid(3, 3, (), (0, 0), (0, 2))

    def test_batch_continues_after_unfinished_and_intermediate_valid(self):
        start, middle, goal = (0, 0), (0, 1), (0, 2)
        grid = self.grid
        rounds = iter([
            ([step_prompt(grid, start, "right")] * 3 + [step_prompt(grid, start, None)],
             ["invalid | step=2 | reason=goal_not_reached"] * 3
             + ["invalid | step=1 | reason=goal_not_reached"]),
            ([step_prompt(grid, middle, "right"), step_prompt(grid, middle, "down")],
             ["valid", "invalid | step=2 | reason=goal_not_reached"]),
            ([step_prompt(grid, goal, "right")], ["invalid | step=1 | reason=out_of_bounds"]),
        ])
        testcase = self

        class Runner:
            def predict(self, inputs, check):
                expected, replies = next(rounds)
                testcase.assertEqual(inputs, expected)
                return replies

        plans = ["right right right left", "right down", "right", ""]
        self.assertEqual(verify_steps(Runner(), [grid] * len(plans), plans),
                         [execute(grid, plan).label() for plan in plans])
        self.assertEqual(list(rounds), [])

    def test_model_verdict_is_not_replaced_by_the_executor(self):
        grid = self.grid

        class Runner:
            def predict(self, inputs, check):
                return ["valid", "unparseable output"]

        self.assertEqual(verify_steps(Runner(), [grid, grid], ["left", "right"]),
                         ["valid", "unparseable output"])

    def test_empty_stop_and_illegal_eos_token_are_distinct(self):
        grid = self.grid
        stop, = observations(grid, execute(grid, ""))
        invalid, = observations(grid, execute(grid, "<eos>"))
        self.assertEqual((stop["action"], stop["step"], stop["reason"]), (None, 1, "goal_not_reached"))
        self.assertEqual((invalid["action"], invalid["step"], invalid["reason"]),
                         ("<eos>", 1, "invalid_action_token"))
