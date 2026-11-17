import unittest
from ppnl.core import Grid
from part3.run_newpart1_transfer import feedback_rule


class PositiveConfirmationTest(unittest.TestCase):
    def test_only_positive_confirmation_is_exposed(self):
        g=Grid(3,3,(),(0,0),(2,0))
        wrong='invalid | step=5 | reason=goal_not_reached'
        self.assertEqual(feedback_rule(g,'down down',wrong,'valid_confirmation'),'valid')
        self.assertEqual(feedback_rule(g,'up','valid','valid_confirmation'),'valid')
        self.assertEqual(feedback_rule(g,'down up','valid','terminal_valid_confirmation'),
                         'invalid | step=3 | reason=goal_not_reached')
        self.assertEqual(feedback_rule(g,'down up',wrong,'eos_index'),
                         'invalid | step=3 | reason=goal_not_reached')
        self.assertEqual(feedback_rule(g,'down down',wrong,'terminal_valid_eos'),'valid')


if __name__=='__main__':unittest.main()
