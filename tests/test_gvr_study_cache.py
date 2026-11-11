import unittest

from ppnl.core import Grid
from scripts.run_gvr_repair_test import CachedGeneration


class GenerationCacheTests(unittest.TestCase):
    def test_one_physical_generation_per_environment(self):
        class Model:
            def __init__(self):
                self.calls = []
                self.last_usage = {}

            def predict(self, task, grid, candidate="", feedback="", check=lambda: None):
                self.calls.append(task)
                self.last_usage = {"input_tokens": 3, "output_tokens": 2}
                return "right right" if task == "generation" else "valid"

        grid = Grid(2, 3, (), (0, 0), (0, 2))
        raw = Model()
        cached = CachedGeneration(raw)
        self.assertEqual(cached.predict("generation", grid), "right right")
        self.assertEqual(cached.predict("generation", grid), "right right")
        self.assertEqual(cached.last_usage, {"input_tokens": 0, "output_tokens": 0})
        self.assertEqual(cached.predict("verification", grid, "right right"), "valid")
        self.assertEqual(raw.calls, ["generation", "verification"])
        self.assertEqual(cached.physical_calls, 1)


if __name__ == "__main__":
    unittest.main()
