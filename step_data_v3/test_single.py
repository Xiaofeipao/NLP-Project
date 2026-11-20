import unittest
from types import SimpleNamespace

from ppnl.core import Grid
from .schema import State
from .single import reference_label, scan, step_input, template_hash, verify_plans


class SingleStepTest(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(3, 3, ((1, 0),), (0, 0), (0, 1))

    def test_first_error_and_stop(self):
        paths = ["", "right", "right left", "right up", "down jump", "STOP", "right left right"]
        expected = ["invalid | step=1 | reason=goal_not_reached", "valid",
                    "invalid | step=3 | reason=goal_not_reached", "invalid | step=2 | reason=out_of_bounds",
                    "invalid | step=1 | reason=obstacle_collision", "invalid | step=1 | reason=invalid_action_token",
                    "valid"]
        self.assertEqual(scan([self.grid]*len(paths), paths,
                              lambda items: [reference_label(s,a) for s,a in items]), expected)

    def test_wrong_model_prediction_is_visible(self):
        self.assertEqual(scan([self.grid], ["down"], lambda items: ["valid"] * len(items)), ["valid"])
        self.assertEqual(scan([self.grid], ["right"], lambda items: ["unknown"]),
                         ["malformed local verdict: unknown"])

    def test_only_local_prompts_reach_model(self):
        g = self.grid
        expected = [step_input(State.from_grid(g), "right"), step_input(State.from_grid(g,g.goal), None)]
        received = []
        def predict(texts, check):
            received.extend(texts)
            return ["valid"]*len(texts)
        config = SimpleNamespace(ppnl_prompt_sha256=template_hash())
        runner = SimpleNamespace(model=SimpleNamespace(config=config), predict=predict)
        self.assertEqual(verify_plans(runner,[g],["right"]), ["valid"])
        self.assertEqual(received, expected)
        self.assertNotEqual(step_input(State.from_grid(g),None),step_input(State.from_grid(g),"STOP"))
        with self.assertRaises(ValueError):
            step_input(State.from_grid(g),"right left")
        with self.assertRaises(ValueError):
            scan([g],[],predict)


if __name__ == "__main__":
    unittest.main()
