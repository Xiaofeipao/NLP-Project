import unittest
from ppnl.core import Grid
from ppnl.verifier_rules import enhance, RULES


class PartialRulesTest(unittest.TestCase):
    def test_rule_scope_and_first_failure(self):
        g=Grid(3,3,((1,0),),(0,0),(2,2))
        self.assertEqual(enhance(g,'down right','valid','terminal'),'valid')
        self.assertEqual(enhance(g,'down right','valid','bounds'),'valid')
        self.assertEqual(enhance(g,'down right','valid','collision'),
                         'invalid | step=1 | reason=obstacle_collision')
        self.assertEqual(enhance(g,'up nonsense','valid','vocabulary'),'valid')
        self.assertEqual(enhance(g,'right','valid','terminal'),
                         'invalid | step=2 | reason=goal_not_reached')
        for rule in RULES:
            self.assertEqual(enhance(g,'right right down down','invalid | step=1 | reason=out_of_bounds',rule),
                             'invalid | step=1 | reason=out_of_bounds')


if __name__=='__main__': unittest.main()
