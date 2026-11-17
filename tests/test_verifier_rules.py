import unittest
from ppnl.core import Grid
from ppnl.verifier_guard import guard_verdict
from part3.run_verifier_rules import run_repair


class VerifierRulesTest(unittest.TestCase):
    def test_terminal_does_not_replace_collision_diagnosis(self):
        grid = Grid(3, 3, (), (0, 0), (2, 0))
        self.assertEqual(guard_verdict(grid, 'up down', 'valid', 'terminal'), 'valid')
        self.assertEqual(guard_verdict(grid, 'down up', 'valid', 'terminal'),
                         'invalid | step=3 | reason=goal_not_reached')
        self.assertEqual(guard_verdict(grid, 'down down', 'valid', 'terminal'), 'valid')
        class NoCalls:
            def scores(self, observations):
                raise AssertionError('Accepted feedback must not trigger R')
        self.assertEqual(run_repair(grid, 'down up', 'valid', NoCalls(),
                                   {'backoff':36}), ('down up', 'accepted'))


if __name__ == '__main__':
    unittest.main()
