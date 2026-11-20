from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from ppnl.core import ACTIONS, execute
from ppnl.data import read_jsonl
from . import g_probes
from .audit_paired_actions import square_orbits, verify_replacements
from .build_paired_actions import write_v_views
from .paired_actions import control_groups, make_rows, row_key
from .paired_steps import verify_group
from .schema import State


def v_row(state, action, row_id):
    return dict(g_probes.example(state,action),id=row_id,mode="v_step",task="verification",
                donor_id="source-"+state.id,parent_id="old-"+row_id,origin="previous_V",shared_g_id=None)


class PairedActionTest(unittest.TestCase):
    def test_previous_v_without_g_has_four_real_independent_cases(self):
        s = State(6,6,((2,5),),(3,5),(3,4))
        old = [v_row(s,"left","existing-v"),v_row(s,"jump","existing-word")]
        current = [v_row(s,a,"v-"+a) for a in ACTIONS]
        expected = dict(left="valid",up="obstacle_collision",right="out_of_bounds",down="goal_not_reached")
        states = {s.id:s}
        with TemporaryDirectory() as temp:
            report,ids = write_v_views(Path(temp),"train",old,current,states,SimpleNamespace(check=execute))
            rows = read_jsonl(Path(temp)/"v_train_four.jsonl")
        self.assertEqual(report["four_independent_moves"],4)
        self.assertEqual({r["candidate"]:r["target"] for r in rows},expected)
        self.assertEqual({r["candidate"] for r in rows if r["in_parent_supervised_V"]},{"left"})
        self.assertTrue(all(r["in_same_split_supervised_V"] and r["source_g_id"] is None for r in rows))
        self.assertTrue(all(r["source_v_ids"]==["existing-v","existing-word"] for r in rows))
        self.assertEqual(set(ids),{(s.id,a) for a in ACTIONS})
        self.assertEqual(len({r["input"].split("Last move:")[0] for r in rows}),1)
        groups = control_groups(current,[],states)
        self.assertEqual({g["kind"] for g in groups},{"four_v_state","same_state_arrival"})
        by_id = {r["id"]:r for r in current}
        for group in groups:
            items = [dict(by_id[i],source={"kind":group["kind"],"group":group["id"],"parent":None})
                     for i in group["rows"]]
            verify_group(items,states)

    def test_replacement_changes_only_v_content_and_dangling_g_anchor(self):
        s = State(6,6,((2,5),),(3,5),(3,4))
        far = State(6,6,(),(0,0),(5,5))
        kept,removed = v_row(s,"left","arrived"),v_row(far,"down","retired")
        g = dict(id="g",mode="g_next",state_id=far.id,anchor_v_id=removed["id"],
                 input="frozen",target="down",optimal_actions=["right","down"])
        removed["shared_g_id"] = g["id"]
        new = make_rows([g,removed,kept],{row_key(kept),(s.id,"down")},{s.id:s,far.id:far})
        self.assertEqual(verify_replacements([g,removed,kept],new),1)
        self.assertIsNone(new[0]["anchor_v_id"])
        self.assertEqual((new[1]["origin"],new[1]["parent_id"],new[1]["target"]),
                         ("V_direction","arrived","goal_not_reached"))
        self.assertEqual(new[1]["input"],g_probes.step_input(s,"down"))
        self.assertEqual(new[2],kept)

    def test_rotated_move_transforms_with_asymmetric_layout(self):
        s = State(6,6,((0,0),(1,3),(2,5)),(3,5),(3,4))
        clockwise = lambda p:(p[1],5-p[0])
        rot = State(6,6,tuple(sorted(map(clockwise,s.obstacles))),clockwise(s.current),clockwise(s.goal))
        self.assertEqual(square_orbits(s,"left"),square_orbits(rot,"up"))
        self.assertNotEqual(square_orbits(s,"left")[1],square_orbits(rot,"down")[1])
        self.assertEqual(square_orbits(s,"jump"),square_orbits(rot,"jump"))
