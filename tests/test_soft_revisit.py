import unittest

from experiments.part1_baseline.policy import episode, run


class SoftRevisitTests(unittest.TestCase):
    def test_soft_revisit_uses_accepted_prefix_and_keeps_g_ranking_by_default(self):
        case = dict(id="loop", grid=dict(start=[1, 1], goal=[1, 3], obstacles=[]),
                    candidate="right")
        prompts = dict(G="{current}", C="{current}:{action}", V="{current}")
        cells = {"(1,1):right": "(1,2)", "(1,2):left": "(1,1)",
                 "(1,2):right": "(1,3)", "(1,2):up": "(0,2)", "(1,2):down": "(2,2)"}

        def predict(task, inputs):
            if task == "G":
                return [["left", "right", "up", "down"] for _ in inputs]
            if task == "C":
                return [cells[text] for text in inputs]
            return ["valid" if text == "(1,3)" else "goal_not_reached" for text in inputs]

        original, soft = run([episode(case, prompts, limit=1),
                              episode(case, prompts, limit=1, soft_revisit=True)], predict)
        self.assertEqual(original["repair_status"], "attempt_limit")
        self.assertEqual(soft["repair_status"], "returned_plan")
        self.assertEqual(soft["repaired_path"], "right right")
        self.assertEqual(len(soft["repair_trace"]), 1)
