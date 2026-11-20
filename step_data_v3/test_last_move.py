import unittest
from types import SimpleNamespace

from ppnl.core import Grid
from . import last_move
from .schema import State


class LastMoveTest(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(6,6,((0,1),(2,3),(3,4)),(0,2),(1,2))
        self.state = State.from_grid(self.grid)

    def test_same_state_separate_actions(self):
        expected = {"down":"valid","right":"goal_not_reached","left":"obstacle_collision",
                    "up":"out_of_bounds","jump":"invalid_action_token"}
        for action,label in expected.items():
            self.assertEqual(last_move.transition(self.state,action)[1],label)
        for value in (None,"","right right"," down"):
            with self.assertRaises(ValueError):
                last_move.step_input(self.state,value)

    def test_online_accepts_on_one_real_move(self):
        calls = []
        def predict(prompts,check):
            calls.extend(prompts)
            return ["valid"]
        runner = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(ppnl_prompt_sha256=last_move.template_hash())),
                                 predict=predict)
        self.assertEqual(last_move.verify_last(runner,self.state,"down"),"valid")
        self.assertEqual(calls,[last_move.step_input(self.state,"down")])

    def test_path_splits_into_states_before_moves(self):
        parts = [(s.current,a) for s,a in last_move.path_steps(self.grid,"right left down")]
        self.assertEqual(parts,[((0,2),"right"),((0,3),"left"),((0,2),"down")])
        self.assertEqual(list(last_move.path_steps(self.grid,"")),[])

    def test_frozen_candidates_count_steps_without_stop(self):
        paths = ["", "down", "right", "down up", "left jump", "right left down"]
        calls = []
        def decide(items):
            calls.extend(a for s,a in items)
            return [last_move.transition(s,a)[1] for s,a in items]
        self.assertEqual(last_move.scan([self.grid]*len(paths),paths,decide),
                         ["invalid | step=1 | reason=goal_not_reached","valid",
                          "invalid | step=2 | reason=goal_not_reached","invalid | step=3 | reason=goal_not_reached",
                          "invalid | step=1 | reason=obstacle_collision","valid"])
        self.assertEqual(len(calls),8)
        self.assertNotIn(None,calls)
        self.assertEqual(last_move.scan([self.grid],["left"],lambda items:["valid"]),["valid"])


if __name__=="__main__":
    unittest.main()
