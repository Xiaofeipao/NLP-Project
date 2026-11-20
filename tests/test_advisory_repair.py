import unittest
from ppnl.core import Grid
from ppnl.advisory_repair import repair_batch


class AdvisoryRepairTest(unittest.TestCase):
    def test_guidance_is_used_with_both_rollback_distances(self):
        class NoCalls:
            def scores(self,obs):raise AssertionError('raw policy must not be called')
        class Guided:
            def __init__(self):self.positions=[]
            def scores(self,obs):
                self.positions.extend(o[1] for o in obs)
                return [[9,8,0,0] for _ in obs]
        g=Grid(3,3,(),(0,0),(2,2))
        for k,pos in [(0,(0,1)),(36,(0,0))]:
            guided=Guided()
            repair_batch([g],['right'],['invalid | step=2 | reason=goal_not_reached'],NoCalls(),guided,backoff=k,prompt_rules=True)
            self.assertEqual(guided.positions,[pos])
    def test_model_action_is_not_filtered_or_replaced(self):
        class Base:
            def scores(self,obs):return [[9,8,0,0] for _ in obs] # illegal up beats legal down
        class Advisory:
            def scores(self,obs):return [[9,8,0,0] for _ in obs]
        g=Grid(3,3,(),(0,0),(2,0))
        for enhanced in [False,True]:
            plans,status=repair_batch([g],[''],['invalid | step=1 | reason=goal_not_reached'],Base(),Advisory(),enhanced)
            self.assertEqual(plans,['up'])
            self.assertEqual(status,['model_illegal_action'])
    def test_valid_prediction_never_triggers_repair(self):
        class NoCalls:
            def scores(self,obs):raise AssertionError('must not invoke policy')
        g=Grid(3,3,(),(0,0),(2,0))
        plans,status=repair_batch([g],['up'],['valid'],NoCalls(),NoCalls(),True)
        self.assertEqual((plans,status),(['up'],['accepted']))


if __name__=='__main__':unittest.main()
