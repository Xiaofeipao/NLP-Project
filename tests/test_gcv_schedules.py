import ast
from collections import Counter
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT/"experiments/path_text_gv"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT))
from gcv_rollout import run_paths
from gcv_schedule import SCHEDULES, make_streams, schedule_batches
from ppnl.core import ACTIONS, Grid, execute
from ppnl.policy_v3_data import distances_to_goal, optimal_moves
from texts import jsonl, point


def reference_responses(task, texts):
    answers = []
    for text in texts:
        fields = {line.split(": ",1)[0]:line.split(": ",1)[1] for line in text.splitlines() if ": " in line}
        p = fields["Current"]
        if p != "no_coordinate":
            p = ast.literal_eval(p)
        if task == "C":
            delta = ACTIONS.get(fields["Action"])
            answers.append(point((p[0]+delta[0],p[1]+delta[1])) if delta else "no_coordinate")
            continue
        goal, walls = ast.literal_eval(fields["Goal"]), tuple(ast.literal_eval(fields["Walls"]))
        if task == "G":
            grid = Grid(6,6,walls,p,goal)
            best = optimal_moves(grid,p,distances_to_goal(grid))
            answers.append(best + [a for a in ACTIONS if a not in best])
        elif p == "no_coordinate":
            answers.append("invalid_action_token")
        elif not all(0 <= n <= 5 for n in p):
            answers.append("out_of_bounds")
        elif p in walls:
            answers.append("obstacle_collision")
        else:
            answers.append("valid" if p == goal else "goal_not_reached")
    return answers


class GCVScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.templates = {t:(CODE/"prompts/coordinate_gcv"/(t+".txt")).read_text().strip() for t in "GCV"}

    @unittest.skipUnless((ROOT/"data/coordinate_gcv_review/train.jsonl").is_file(), "requires the prepared G/C/V corpus")
    def test_all_arms_share_complete_streams_and_preserve_stages(self):
        rows = jsonl(ROOT/"data/coordinate_gcv_review/train.jsonl")
        streams, weights, summary = make_streams(rows)
        self.assertEqual(len(rows), len(weights))
        self.assertEqual(summary["views"], {"C":16400,"GV":40800})
        for arm in SCHEDULES:
            steps = schedule_batches(arm,streams)
            self.assertEqual(len(steps),375)
            uses=Counter(i for _,_,batch in steps for i in batch)
            self.assertTrue(all(uses[i] == (1 if row["task"]=="C" else 2) for i,row in enumerate(rows)))
            for stream in streams:
                self.assertEqual([b for t,_,b in steps if t==stream],streams[stream])
        self.assertEqual(SCHEDULES["uniform"], ("C","GV","GV")*125)
        self.assertEqual(SCHEDULES["short_warmup"], ("C",)*25+("C","GV","GV")*100+("GV",)*50)
        self.assertEqual(SCHEDULES["c_first"], ("C",)*125+("GV",)*250)

    @unittest.skipUnless(all((ROOT/"data/coordinate_gcv_review"/(p+"_inputs.jsonl")).is_file() and
                             (ROOT/"data/random_errors_20261123"/(p+"_error_200.jsonl")).is_file()
                             for p in ("single", "double")), "requires both frozen path corpora")
    def test_oracle_fixture_recovers_both_sets_and_repairs_from_detected_prefix(self):
        for split in ("single","double"):
            inputs = jsonl(ROOT/"data/coordinate_gcv_review"/(split+"_inputs.jsonl"))
            gold = jsonl(ROOT/"data/random_errors_20261123"/(split+"_error_200.jsonl"))
            pred = run_paths(inputs,self.templates,reference_responses)
            self.assertEqual(len(pred),200)
            for src,ref,r in zip(inputs,gold,pred,strict=True):
                self.assertEqual(r["verification"],dict(label="invalid",step=ref["step"],reason=ref["reason"]))
                self.assertIn(r["repair_status"], ("returned_prefix","returned_plan"))
                self.assertTrue(execute(Grid.from_dict(src["grid"]),r["repaired_path"]).valid)
            if split == "single":
                self.assertEqual(sum(r["repair_status"]=="returned_prefix" for r in pred),29)

    def test_model_coordinate_errors_propagate_without_teacher_forcing(self):
        case = dict(id="one",grid=dict(start=[0,3],goal=[0,4],obstacles=[[0,2]]),candidate="left")
        def wrong(task,texts):
            if task=="C":
                return ["(0,4)"]*len(texts)
            return reference_responses(task,texts)
        result=run_paths([case],self.templates,wrong)[0]
        self.assertEqual(result["verification"]["label"],"valid")
        self.assertEqual(result["repaired_path"],"left")
        result=run_paths([case],self.templates,lambda task,texts:["broken"]*len(texts))[0]
        self.assertEqual(result["repair_status"],"malformed_C")


if __name__ == "__main__":
    unittest.main()
