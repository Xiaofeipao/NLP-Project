import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from reachability_probe import load_cases, reward, score
from run_tree_verifier import events, predict
from ppnl.core import Grid


class ReachabilityExperiments(unittest.TestCase):
    def test_paired_maps_and_false_refusal(self):
        grids, _, answers = load_cases()
        self.assertEqual(score(grids,answers)['reachable_solved'],4)
        refusal = {key:'NO_PATH' for key in grids}
        self.assertEqual(score(grids,refusal)['reachable_false_refusals'],4)
        self.assertEqual(sum(reward(g,'NO_PATH') for g in grids.values()),0)

    def test_literal_eos_is_an_invalid_action(self):
        grid=Grid(6,6,(),(0,0),(5,5))
        rows=list(events(grid,'<eos>'))
        self.assertEqual(rows[0][1][13],1)
        self.assertEqual(rows[0][1][12],0)
        self.assertEqual(rows[1][1][12],1)

    def test_prediction_does_not_call_truth_executor(self):
        class Tree:
            def predict(self, rows): return ['obstacle_collision']
        grid=Grid(6,6,((0,1),),(0,0),(5,5))
        with patch('run_tree_verifier.execute',side_effect=AssertionError('label leakage')):
            self.assertEqual(predict(Tree(),grid,'right'),'invalid | step=1 | reason=obstacle_collision')


if __name__=='__main__': unittest.main()
