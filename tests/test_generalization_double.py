import json
from pathlib import Path
import tempfile
import unittest
from scripts.build_generalization_double import build
from ppnl.core import Grid,execute
from ppnl.data import read_jsonl,edit_distance


class DoubleExtensionTest(unittest.TestCase):
    def test_pairing_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'double';build(output)
            for name in ['7x7','dense_6x6']:
                old=read_jsonl(f'data/generalization_v1/test_{name}.jsonl')
                new=read_jsonl(output/f'test_{name}_double.jsonl')
                frozen=Path(f'data/generalization_double/test_{name}_double.jsonl')
                self.assertEqual(frozen.read_bytes(),(output/frozen.name).read_bytes())
                for a,b in zip(old,new):
                    self.assertEqual(a['grid'],b['grid'])
                    self.assertEqual(a['id'],b['parent_id'])
                    self.assertEqual(execute(Grid.from_dict(b['grid']),b['candidate']).label(),b['label'])
                    if a['valid']:self.assertEqual(a['candidate'],b['candidate'])
                    else:
                        self.assertEqual(len(b['edits']),2)
                        self.assertEqual(edit_distance(b['target_plan'].split(),b['candidate'].split()),2)


if __name__=='__main__':unittest.main()
