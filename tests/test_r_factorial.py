import unittest
from part3.run_r_factorial import variants,choose
from ppnl.repair_variants import repair
from ppnl.core import Grid,execute


class FactorialTest(unittest.TestCase):
    def test_eight_independent_switches(self):
        specs=variants()
        self.assertEqual(len(specs),8)
        self.assertEqual(specs['T1P1E1'],{'mode':'beam_soft','score_cap':108,'erase':True})
        class Scorer:
            def scores(self,obs):return [[0,4,1,2] for _ in obs]
        g=Grid(3,3,(),(0,0),(2,0))
        for spec in specs.values():
            out,_,_=repair(g,'','invalid | step=1 | reason=goal_not_reached',Scorer(),spec)
            self.assertTrue(execute(g,out).valid)
        base={'success_rate':1,'optimality':1,'score_rows':2}
        self.assertEqual(choose({n:base for n in specs}),'T0P0E0')


if __name__=='__main__':unittest.main()
