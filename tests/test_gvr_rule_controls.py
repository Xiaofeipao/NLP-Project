import unittest

from ppnl.core import Grid, execute
from ppnl.gvr_rule_controls import rule_repair


class RuleControlTests(unittest.TestCase):
    def test_controls_use_same_decoder_and_return_executable_paths(self):
        grid = Grid(3, 3, (), (0, 0), (2, 2))
        for kind in ("uniform", "manhattan"):
            with self.subTest(kind=kind):
                result = rule_repair(grid, kind)
                self.assertTrue(result.completed)
                self.assertTrue(execute(grid, result.plan).valid)
                self.assertLessEqual(result.decoder_row_steps, 148)


if __name__ == "__main__":
    unittest.main()
