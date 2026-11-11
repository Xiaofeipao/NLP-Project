import unittest
from scripts.run_full_finetune_gvr import repair_spec

class RepairOptions(unittest.TestCase):
    def test_default_is_ordinary(self):
        self.assertIsNone(repair_spec({'mode':'executor_one'}))

    def test_constraints_are_explicit(self):
        spec=repair_spec({'mode':'executor_one','repair_decoding':'constrained'})
        self.assertEqual((spec.beam_width,spec.max_actions,spec.state_constraints),(4,36,True))

    def test_reject_self_and_direct_constraints(self):
        for mode in ('self','direct'):
            with self.assertRaises(ValueError):
                repair_spec({'mode':mode,'repair_decoding':'constrained'})

    def test_unknown_mode_and_invalid_budget(self):
        with self.assertRaises(ValueError):
            repair_spec({'mode':'executor_one','repair_decoding':'bad'})
        with self.assertRaises(ValueError):
            repair_spec({'mode':'executor_one','repair_decoding':'constrained','beam_width':0})
