import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_reachability_cot import answer, target
from reachability_probe import load_cases, reward

class CotContract(unittest.TestCase):
    def test_targets_and_strict_delimiter(self):
        for grid in load_cases()[0].values():
            self.assertEqual(reward(grid,answer(target(grid))),1)
        self.assertEqual(answer('NO_PATH'),'')
        self.assertEqual(answer('FINAL: NO_PATH FINAL: right'),'')
        self.assertEqual(answer('trace 0,0 FINAL: NO_PATH'),'NO_PATH')

if __name__=='__main__': unittest.main()
