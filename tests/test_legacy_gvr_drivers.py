"""Retired batch commands must stop before loading a model or touching CUDA."""
import builtins
import hashlib
import json
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DRIVERS = ("smoke_gvr_repair.py", "run_gvr_development.py",
           "run_gvr_repair_test.py", "run_gvr_optional_development.py")


class LegacyGVRDriverTests(unittest.TestCase):
    def test_retired_entry_points_fail_before_model_imports(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "ppnl.modeling" or name.split(".")[0] in ("torch", "transformers"):
                self.fail(f"Retired entry point tried to load {name}")
            return original_import(name, *args, **kwargs)

        for name in DRIVERS:
            with self.subTest(driver=name), patch("builtins.__import__", guarded_import):
                with patch.object(sys, "argv", [name, "--help"]):
                    with self.assertRaises(SystemExit) as stopped:
                        runpy.run_path(str(ROOT / "scripts" / name), run_name="__main__")
                self.assertIn("run_post_training.py", str(stopped.exception))
                self.assertIn("23183e2", str(stopped.exception))

    def test_historical_archive_matches_its_sha_manifest(self):
        archive = ROOT / "docs/legacy_gvr_drivers_23183e2.zip"
        digest, name = archive.with_suffix(".sha256").read_text(encoding="ascii").split()
        self.assertEqual(name, archive.name)
        self.assertEqual(digest, hashlib.sha256(archive.read_bytes()).hexdigest())
        with zipfile.ZipFile(archive) as source:
            manifest = json.loads(source.read("MANIFEST.json"))
            self.assertEqual(manifest["revision"], "23183e22f6d9ec0067c30101dad9f35cab07b9e9")
            self.assertEqual(set(manifest["files"]),
                             {"scripts/" + name for name in (*DRIVERS, "freeze_gvr_repair.py")})
            for name, expected in manifest["files"].items():
                data = source.read(name)
                self.assertEqual(len(data), expected["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected["sha256"])


if __name__ == "__main__":
    unittest.main()
