from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from ppnl.core import Grid
from ppnl.prompts import (prompt, format_training_rows, protocol_model_config,
                          check_checkpoint_prompt_format)


class PromptFormatsTests(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(6, 6, ((1, 3),), (1, 1), (1, 4))

    def test_default_is_exact_historical_template(self):
        self.assertEqual(prompt("generation", self.grid),
                         f"Plan a path. {self.grid.description()}\nReturn only space-separated up down left right actions.")
        self.assertEqual(prompt("repair", self.grid, "right", "invalid"),
                         f"Repair a path. {self.grid.description()}\nCandidate: right\nFeedback: invalid\n"
                         "Return only the corrected space-separated up down left right actions.")

    def test_reformat_does_not_mutate_ids_targets_or_source(self):
        rows = [{"task": "generation", "id": "g", "grid": asdict(self.grid),
                 "input": prompt("generation", self.grid), "target": "SECRET_REFERENCE", "source_id": "fixture"}]
        before = json.dumps(rows)
        changed = format_training_rows(rows, "structured")
        self.assertEqual(json.dumps(rows), before)
        self.assertEqual(changed[0]["target"], "SECRET_REFERENCE")
        self.assertEqual(changed[0]["id"], "g")
        self.assertNotIn("SECRET_REFERENCE", changed[0]["input"])
        self.assertIn("Grid: 6x6", changed[0]["input"])

    def test_verification_never_receives_gold_diagnosis(self):
        source = prompt("verification", self.grid, "right", "SECRET_LABEL", input_format="structured")
        self.assertIn("Candidate: right", source)
        self.assertNotIn("SECRET_LABEL", source)
        repair = prompt("repair", self.grid, "right", "MODEL_FEEDBACK", input_format="structured")
        self.assertIn("Feedback: MODEL_FEEDBACK", repair)

    def test_checkpoint_format_mismatch_rejected_without_ml(self):
        with tempfile.TemporaryDirectory() as temp:
            check_checkpoint_prompt_format(temp, "vanilla")
            with self.assertRaisesRegex(ValueError, "Checkpoint prompt format"):
                check_checkpoint_prompt_format(temp, "structured")
            (Path(temp) / "prompt_format.json").write_text(json.dumps({"prompt_format": "structured"}))
            check_checkpoint_prompt_format(temp, "structured")
            with self.assertRaises(ValueError):
                check_checkpoint_prompt_format(temp, "vanilla")

    def test_unknown_and_conflicting_formats_rejected(self):
        with self.assertRaises(ValueError):
            prompt("generation", self.grid, input_format="cot")
        with self.assertRaises(ValueError):
            protocol_model_config({"prompt_format": "vanilla"}, {"prompt_format": "structured"})
        self.assertEqual(protocol_model_config({}, {})["prompt_format"], "vanilla")
