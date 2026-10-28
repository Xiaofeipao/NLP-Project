from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from ppnl.core import Grid, execute
from ppnl.evaluation import evaluate


class ScriptedRunner:
    def __init__(self):
        self.responses = iter([
            ["valid", "invalid | step=1 | reason=out_of_bounds"],
            ["right right"], ["up"], ["invalid | step=1 | reason=out_of_bounds"],
            ["right right"], ["right right"], ["right right"],
        ])

    def predict(self, inputs, check):
        check()
        if not inputs:
            return []
        values = next(self.responses)
        if len(values) != len(inputs):
            raise AssertionError("Unexpected evaluator call")
        return values


class Oracle:
    sha256 = "test-executor"

    def check(self, grid, plan):
        return execute(grid, plan)


class PipelineTests(unittest.TestCase):
    def test_full_pipeline_repairs_and_stops_on_success(self):
        grid = Grid(2, 3, (), (0, 0), (0, 2))
        rows = []
        for i, candidate in enumerate(("right right", "up")):
            v = execute(grid, candidate)
            rows.append({"id": str(i), "source_id": "fixture", "layout_id": grid.layout_id,
                         "grid": asdict(grid), "candidate": candidate, "valid": v.valid,
                         "reason": v.reason, "step": v.step, "label": v.label()})
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            out = Path(temp) / "evaluation"
            metrics = evaluate(ScriptedRunner(), rows, out, full=True, rounds=3, oracle=Oracle())
            self.assertEqual(metrics["verification"]["macro_f1"], 1.0)
            self.assertEqual(metrics["direct_generation"]["success_rate"], 0.0)
            self.assertEqual(metrics["self_verification_repair"]["success_rate"], 1.0)
            self.assertEqual(metrics["improved_iterative_repair"]["success_rate"], 1.0)
            self.assertEqual(metrics["ablation_one_round"]["success_rate"], 1.0)
            traces = [json.loads(line) for line in (out / "pipeline_traces.jsonl").read_text().splitlines()]
            self.assertEqual(len(traces[0]["iterations"]), 1)
            self.assertEqual(len(json.loads((out / "case_studies.json").read_text())), 1)


if __name__ == "__main__":
    unittest.main()
