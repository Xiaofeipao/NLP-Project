"""Meaningful drift/leakage regression checks; no model or CUDA loading."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from ppnl.core import Grid
from ppnl.experiment_freeze import (audit_splits, create_freeze, file_hash,
                                    validate_freeze, write_new_json)
from ppnl.task_adapters import checkpoint_hash


class ExperimentFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        (self.root / "ppnl").mkdir()
        (self.root / "ppnl/example.py").write_text("VERSION = 1\n")
        executor = self.root / "vendor/ppnl/executor-point-sg.py"
        executor.parent.mkdir(parents=True)
        executor.write_text("# test executor\n")
        self.rows = {}
        layout_splits = {}
        for n, split in enumerate(("train", "dev", "test_id", "test_ood")):
            grid = Grid(6, 6, ((0, min(n, 2)),), (1, 1), (2, 2))
            tasks = ("generation", "verification", "repair") if n < 2 else (None,)
            self.rows[split] = [{"id": f"{split}-{i}", "grid": {
                "rows": 6, "cols": 6, "obstacles": list(grid.obstacles),
                "start": grid.start, "goal": grid.goal}, "layout_id": grid.layout_id,
                "source_id": f"source-{min(n, 2)}-{i}", **({"task": task} if task else {})}
                for i, task in enumerate(tasks)]
            layout_splits[split if n < 2 else "test"] = [grid.layout_id]
        manifest = {"seed": 42, "upstream_revision": "fixture", "layout_splits": layout_splits,
                    "executor_sha256": file_hash(executor), "counts": {
                        "train": dict.fromkeys(("generation", "verification", "repair"), 1),
                        "dev": dict.fromkeys(("generation", "verification", "repair"), 1),
                        "test_id": 1, "test_ood": 1}}
        (self.data / "manifest.json").write_text(json.dumps(manifest))
        self.save_rows()
        self.protocol = {"dataset_root": str(self.data)}
        for name, key in (("generation", "generation_base"), ("verification", "verification_adapter"),
                          ("repair", "repair_adapter")):
            directory = self.root / name
            directory.mkdir()
            (directory / "model.safetensors").write_bytes(name.encode())
            (directory / "config.json").write_text("{}")
            self.protocol[key] = str(directory)
        self.config = {"seed": 42, "training": {"max_target_tokens": 128}}
        self.report = self.root / "verify_dev.json"
        self.report.write_text(json.dumps({"disabled_adapter_matches_base": True,
            "dev_diagnostics": {"generation": {"success_rate": 0}}, "identity": {
                "base_sha256": checkpoint_hash(self.protocol["generation_base"]),
                "adapters_sha256": {name: checkpoint_hash(self.protocol[f"{name}_adapter"])
                                    for name in ("verification", "repair")}}}))

    def save_rows(self):
        for split, rows in self.rows.items():
            (self.data / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def lock(self):
        record = create_freeze(self.config, self.protocol, self.report, "Selected on dev", self.root)
        path = self.root / "freeze.json"
        write_new_json(path, record)
        return path

    def test_paired_tests_allowed_and_exact_ids_exported(self):
        audit = audit_splits(self.data)
        self.assertEqual(audit["splits"]["train"]["ids"], [r["id"] for r in self.rows["train"]])
        self.assertEqual(audit["test_status"], "reused_exploratory")
        self.assertEqual(validate_freeze(self.lock(), self.config, self.protocol, self.root)["test_status"],
                         "reused_exploratory")

    def test_development_report_on_external_output_disk_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as external:
            external_report = Path(external) / "verify_dev.json"
            external_report.write_bytes(self.report.read_bytes())
            record = create_freeze(self.config, self.protocol, external_report, "Dev selection", self.root)
            lock = self.root / "external_freeze.json"
            write_new_json(lock, record)
            self.assertEqual(validate_freeze(lock, self.config, self.protocol, self.root)["test_status"], "reused_exploratory")
            external_report.write_text("{}")
            with self.assertRaises(ValueError): validate_freeze(lock, self.config, self.protocol, self.root)

    def test_cross_split_ids_rejected(self):
        self.rows["dev"][0]["id"] = self.rows["train"][0]["id"]
        self.save_rows()
        with self.assertRaisesRegex(ValueError, "ID leakage"):
            audit_splits(self.data)

    def test_layout_leakage_rejected_even_with_different_ids(self):
        self.rows["dev"][0].update({key: self.rows["train"][0][key] for key in ("grid", "layout_id")})
        self.save_rows()
        with self.assertRaisesRegex(ValueError, "Layout leakage"):
            audit_splits(self.data)

    def test_unpaired_test_environments_rejected(self):
        self.rows["test_ood"][0]["grid"]["goal"] = (3, 3)
        self.save_rows()
        with self.assertRaisesRegex(ValueError, "paired"):
            audit_splits(self.data)

    def test_config_and_tokenizer_and_code_drift_rejected(self):
        lock = self.lock()
        changed = copy.deepcopy(self.config)
        changed["training"]["max_target_tokens"] = 64
        with self.assertRaisesRegex(ValueError, "config_sha256"):
            validate_freeze(lock, changed, self.protocol, self.root)
        tokenizer = Path(self.protocol["generation_base"]) / "tokenizer.json"
        tokenizer.write_text("changed")
        with self.assertRaisesRegex(ValueError, "models"):
            validate_freeze(lock, self.config, self.protocol, self.root)
        tokenizer.unlink()
        (self.root / "ppnl/example.py").write_text("VERSION = 2\n")
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            validate_freeze(lock, self.config, self.protocol, self.root)

    def test_wrong_weight_identity_rejected(self):
        (Path(self.protocol["repair_adapter"]) / "model.safetensors").write_bytes(b"other")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.lock()

    def test_dataset_bytes_drift_rejected(self):
        lock = self.lock()
        self.rows["test_id"][0]["candidate"] = "left"
        self.save_rows()
        with self.assertRaisesRegex(ValueError, "split_audit"):
            validate_freeze(lock, self.config, self.protocol, self.root)

    def test_test_entry_requires_freeze_before_cuda(self):
        from unittest.mock import patch
        from scripts.run_gvr import main
        protocol_path = self.root / "protocol.json"
        config_path = self.root / "config.json"
        protocol_path.write_text(json.dumps(self.protocol))
        config_path.write_text(json.dumps(self.config))
        argv = ["run_gvr.py", "evaluate", "--protocol", str(protocol_path), "--config", str(config_path),
                "--split", "test_id", "--out", str(self.root / "new_eval")]
        # evaluation_output needs the protocol default even with an explicit --out.
        self.protocol["output_root"] = str(self.root / "outputs")
        protocol_path.write_text(json.dumps(self.protocol))
        with patch("sys.argv", argv), patch("ppnl.modeling.require_cuda") as cuda, \
                patch("ppnl.task_adapters.GVRModel") as model, patch("sys.stderr"):
            with self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            cuda.assert_not_called()
            model.assert_not_called()
        self.assertFalse((self.root / "new_eval").exists())

    def test_test_metrics_cannot_replace_development_verification(self):
        self.report.write_text(json.dumps({"verification": {"macro_f1": 1}}))
        with self.assertRaisesRegex(ValueError, "verify_dev"):
            self.lock()

    def test_lock_cannot_be_overwritten(self):
        lock = self.lock()
        before = lock.read_bytes()
        with self.assertRaises(FileExistsError):
            write_new_json(lock, {"replacement": True})
        self.assertEqual(before, lock.read_bytes())


if __name__ == "__main__":
    unittest.main()
