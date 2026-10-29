import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ppnl.harness import _atomic_json


class AtomicSnapshotTests(unittest.TestCase):
    def test_temporary_windows_permission_failure_is_retried(self):
        import os
        real_replace = os.replace
        attempts = []

        def replace(source, target):
            attempts.append(1)
            if len(attempts) < 3:
                raise PermissionError('snapshot is briefly open')
            real_replace(source, target)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'state.json'
            path.write_text('{"old": true}')
            with patch('ppnl.harness.os.replace', side_effect=replace), patch('ppnl.harness.time.sleep'):
                _atomic_json(path, {'new': True})
            self.assertEqual(json.loads(path.read_text()), {'new': True})
            self.assertEqual(len(attempts), 3)

    def test_persistent_failure_preserves_original_and_temp(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'state.json'
            path.write_text('{"old": true}')
            with patch('ppnl.harness.os.replace', side_effect=PermissionError('denied')) as replace, \
                    patch('ppnl.harness.time.sleep'):
                with self.assertRaises(PermissionError):
                    _atomic_json(path, {'new': True})
            self.assertEqual(replace.call_count, 6)
            self.assertEqual(json.loads(path.read_text()), {'old': True})
            self.assertEqual(json.loads(path.with_suffix('.json.tmp').read_text()), {'new': True})

    def test_unrelated_io_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch('ppnl.harness.os.replace', side_effect=OSError('storage failed')) as replace, \
                    patch('ppnl.harness.time.sleep') as sleep:
                with self.assertRaises(OSError):
                    _atomic_json(Path(temp) / 'state.json', {})
            self.assertEqual(replace.call_count, 1)
            sleep.assert_not_called()
