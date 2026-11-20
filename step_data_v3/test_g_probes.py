import unittest
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

from ppnl.core import ACTIONS, Grid
from . import g_probes, last_move
from .balance_steps import dual_axis_arrival, shortest_final_moves
from .schema import State
from .single import LABELS


class GProbeTest(unittest.TestCase):
    def test_batch_loss_equal_decision_weight_for_each_v_label(self):
        import torch
        from transformers import BatchEncoding
        from ppnl.modeling import Seq2Seq

        # Compact token IDs for the five actual target lengths (including EOS).
        lengths = (2, 6, 7, 7, 7)
        labels = torch.full((6, 7), -100, dtype=torch.long)
        for i, length in enumerate(lengths):
            labels[i, 0] = i
            labels[i, 1:length] = 7
        labels[5, :2] = torch.tensor([1, 7])
        allowed = torch.zeros(6, 4, dtype=torch.bool)
        allowed[5, [1, 3]] = True
        logits = torch.zeros(6, 7, 8, requires_grad=True)
        batch = BatchEncoding({"labels": labels, "policy_actions": allowed})
        runner = Seq2Seq.__new__(Seq2Seq)
        runner.device = "cpu"
        runner.policy_loss = "optimal_actions"
        runner.action_ids = [0, 1, 2, 3]
        runner.model = lambda **inputs: SimpleNamespace(logits=logits)
        g_probes.batch_loss(runner, batch).backward()

        # For uniform logits, undo the batch mean and the derivative of cross-entropy.
        scale = len(labels) / (1 - 1 / logits.shape[-1])
        for i, length in enumerate(lengths):
            self.assertAlmostEqual(float(-logits.grad[i, 0, i]) * scale, 0.5, places=6)
            suffix = -logits.grad[i, 1:length, 7] * scale
            torch.testing.assert_close(suffix, torch.full_like(suffix, 0.5 / (length - 1)))
            self.assertTrue(logits.grad[i, length:].eq(0).all())
        torch.testing.assert_close(logits.grad[5, 0, :4] * len(labels),
                                   torch.tensor([0.25, -0.25, 0.25, -0.25]))
        self.assertTrue(logits.grad[5, 1:].eq(0).all())

    def test_four_independent_questions_keep_all_optimal_actions(self):
        state = State.from_grid(Grid(3,3,(),(0,0),(1,1)))
        g = {"mode":"g_next","id":"g0","donor_id":"source0","state_id":state.id,
             "target":"right","optimal_actions":["right","down"]}
        cases = list(g_probes.g_examples([g],{state.id:state}))
        self.assertEqual(len({r["input"] for r in cases}),4)
        self.assertEqual({r["candidate"] for r in cases},set(ACTIONS))
        self.assertEqual([r["candidate"] for r in cases if r["is_G_representative"]],["right"])
        self.assertEqual({r["candidate"] for r in cases if r["is_G_optimal"]},{"right","down"})
        for r in cases:
            self.assertEqual(r["target"],"goal_not_reached" if r["is_G_optimal"] else "out_of_bounds")
            self.assertEqual(r["input"],g_probes.step_input(state,r["candidate"]))
            self.assertNotIn("source0",r["input"])

    def test_shared_state_exercises_all_labels_in_the_same_template(self):
        s = State.from_grid(Grid(6,6,((0,1),),(0,2),(1,2)))
        observations = {a:g_probes.example(s,a) for a in (*ACTIONS,"jump")}
        self.assertEqual({a:r["target"] for a,r in observations.items()},
                         dict(up="out_of_bounds",down="valid",left="obstacle_collision",
                              right="goal_not_reached",jump="invalid_action_token"))
        self.assertEqual(len({tuple(x for x in r["input"].splitlines() if not x.startswith("Last move:"))
                              for r in observations.values()}),1)
        for a in (None,"","right right","right\n"," up"):
            with self.assertRaises(ValueError):
                g_probes.step_input(s,a)

    def test_online_and_fixed_candidate_use_the_same_model_entry(self):
        grid = Grid(3,3,((0,1),),(0,0),(1,0))
        s = State.from_grid(grid)
        seen = []
        def predict(texts,check):
            seen.extend(texts)
            return ["valid" if '"down"' in x else "goal_not_reached" for x in texts]
        config = SimpleNamespace(ppnl_prompt_sha256=g_probes.template_hash())
        runner = SimpleNamespace(model=SimpleNamespace(config=config),predict=predict)
        self.assertEqual(g_probes.verify_last(runner,s,"down"),"valid")
        self.assertEqual(g_probes.verify_plans(runner,[grid]*3,["","down","down up"]),
                         ["invalid | step=1 | reason=goal_not_reached","valid","invalid | step=3 | reason=goal_not_reached"])
        self.assertEqual(seen,[g_probes.step_input(s,"down")]*3+
                         [g_probes.step_input(State.from_grid(grid,grid.goal),"up")])
        config.ppnl_prompt_sha256 = last_move.template_hash()
        with self.assertRaises(ValueError):
            g_probes.verify_last(runner,s,"down")

    def test_balanced_view_samples_distinct_prompts(self):
        rows = [{"id":f"{label}{i}","input":f"{label}{i}","target":label}
                for label in LABELS for i in range(2 if label==LABELS[0] else 3)]
        repeated = rows + [dict(rows[0],id="same-input")]
        selected = g_probes.balanced_rows(repeated,42)
        self.assertEqual(selected,g_probes.balanced_rows(repeated,42))
        self.assertEqual(Counter(r["target"] for r in selected),{c:2 for c in LABELS})
        self.assertEqual(len({r["input"] for r in selected}),10)
        with self.assertRaisesRegex(ValueError,"conflicting targets"):
            g_probes.balanced_rows(rows+[dict(rows[0],target=LABELS[-1])],42)

    def test_shortest_arrival_condition_uses_the_original_start_and_walls(self):
        grid = Grid(3,3,(),(0,0),(1,1))
        self.assertEqual(shortest_final_moves(grid),{"down":(0,1),"right":(1,0)})
        self.assertTrue(dual_axis_arrival(grid,State.from_grid(grid,(1,0)),"right"))
        self.assertFalse(dual_axis_arrival(grid,State.from_grid(grid,(2,1)),"up"))
        wall = replace(grid,obstacles=((1,0),))
        self.assertEqual(shortest_final_moves(wall),{"down":(0,1)})
        self.assertFalse(dual_axis_arrival(wall,State.from_grid(wall,(0,1)),"down"))


if __name__=="__main__":
    unittest.main()
