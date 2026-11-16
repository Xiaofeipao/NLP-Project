import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from part2.workflow import verify_results
from ppnl.method_sweep import METHODS


class PartsReleaseTest(unittest.TestCase):
    def test_part3_protocol_matches_methods(self):
        p = Path(__file__).resolve().parents[1] / "part3/protocol.json"
        self.assertEqual(json.loads(p.read_text())["methods"], METHODS)

    def test_changed_evidence_is_rejected_before_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "corrupt.zip"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("results.json", "changed")
                z.writestr(
                    "PUBLIC_EXPORT.json",
                    json.dumps(
                        {
                            "files": {
                                "results.json": {
                                    "export_sha256": hashlib.sha256(b"original").hexdigest()
                                }
                            }
                        }
                    ),
                )
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_results(p, 2)
