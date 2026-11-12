"""CPU-only integrity and reproducibility check for optional evaluation data."""
import json
from pathlib import Path
import tempfile
import unittest

from ppnl.core import Grid, OfficialOracle, shortest_path
from ppnl.data import read_jsonl
from scripts.build_generalization_dataset import ROOT, build, digest


class GeneralizationDatasetTest(unittest.TestCase):
    def test_reproducibility_truth_and_isolation(self):
        published = ROOT / 'data/generalization_v1'
        manifest = json.loads((published / 'manifest.json').read_text())
        for name, expected in manifest['source_sha256'].items():
            self.assertEqual(digest(ROOT / name), expected, name)
        self.assertEqual(digest(ROOT / 'scripts/build_generalization_dataset.py'), manifest['generator_sha256'])
        oracle = OfficialOracle(ROOT / 'vendor/ppnl/executor-point-sg.py')
        self.assertEqual(oracle.sha256, manifest['executor_sha256'])
        layouts = set()
        for name in manifest['source_sha256']:
            if name.endswith('.jsonl'):
                layouts.update(r['layout_id'] for r in read_jsonl(ROOT / name))
            else:
                for group in json.loads((ROOT / name).read_text()).get('layout_splits', {}).values():
                    layouts.update(group)
        with tempfile.TemporaryDirectory() as directory:
            rebuilt = Path(directory) / 'dataset'
            build(rebuilt, manifest['seed'])
            for filename, expected in manifest['files_sha256'].items():
                self.assertEqual(digest(published / filename), expected)
                self.assertEqual(digest(rebuilt / filename), expected)
                rows = read_jsonl(rebuilt / filename)
                self.assertEqual(len(rows), 200)
                for row in rows:
                    grid = Grid.from_dict(row['grid'])
                    self.assertEqual(grid.layout_id, row['layout_id'])
                    self.assertNotIn(grid.layout_id, layouts)
                    layouts.add(grid.layout_id)
                    self.assertEqual(' '.join(shortest_path(grid)), row['target_plan'])
                    self.assertTrue(oracle.check(grid, row['target_plan']).valid)
                    verdict = oracle.check(grid, row['candidate'])
                    self.assertEqual(verdict.label(), row['label'])
                    self.assertEqual(verdict.valid, row['valid'])
                    self.assertEqual(verdict.reason, row['reason'])
                    self.assertEqual(verdict.step, row['step'])
                    self.assertEqual(json.loads(json.dumps(verdict.trace)), row['trace'])
                    if filename == 'test_dense_6x6.jsonl':
                        self.assertEqual((grid.rows, grid.cols), (6, 6))
                        self.assertTrue(9 <= len(grid.obstacles) <= 12)
                    else:
                        self.assertEqual((grid.rows, grid.cols), (7, 7))
            with self.assertRaises(FileExistsError):
                build(rebuilt)


if __name__ == '__main__':
    unittest.main()
