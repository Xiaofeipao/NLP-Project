import unittest
from part3.run_feedback_backoff_sweep import cut_repair,select_k
from ppnl.core import Grid


class BackoffSweepTest(unittest.TestCase):
    def test_no_oracle_correction_and_dev_ranking(self):
        class NoCalls:
            def scores(self,obs):raise AssertionError('V-valid must not invoke R')
        g=Grid(3,3,(),(0,0),(2,0))
        self.assertEqual(cut_repair(g,'down up','valid',36,NoCalls(),{},{}),('down up','accepted',0))
        m={str(k):{'success_rate':.8,'optimality':.7,'score_rows':20} for k in range(1,37)}
        self.assertEqual(select_k(m),1)
        m['4']['success_rate']=.9;m['4']['optimality']=.5
        self.assertEqual(select_k(m),4)


if __name__=='__main__':unittest.main()
