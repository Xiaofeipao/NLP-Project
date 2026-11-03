import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ppnl.core import Grid, execute
from ppnl.prefix_state_repair import repair_generated
from ppnl.prefix_evaluation import create_lock, validate_lock, run, prepare
from ppnl.prompts import rlvr_prompt_config, prompt
from ppnl.experiment_freeze import write_new_json


class ReviewRegressions(unittest.TestCase):
    def test_restart_discards_even_successful_or_overlong_prefix(self):
        grid = Grid(2, 3, (), (0, 0), (0, 2))
        class Oracle:
            check = staticmethod(execute)
        class Scores:
            def __init__(self): self.positions = []
            def scores(self, observations):
                self.positions.extend(p for g, p in observations)
                return [[0, 0, 0, 10] for _ in observations]
        for candidate in ('right right up', 'right left right left up', 'right right'):
            scorer = Scores()
            result = repair_generated(grid, candidate, scorer, Oracle(), 'restart_visit', 2)
            self.assertEqual(result['kept_prefix'], '')
            self.assertEqual(scorer.positions, [(0, 0), (0, 1)])
            self.assertEqual(result['plan'], 'right right')
        result = repair_generated(grid, 'right left right right', Scores(), Oracle(), max_steps=2)
        self.assertEqual(result['plan'], 'right left')
        self.assertEqual(result['stop_reason'], 'prefix_exceeds_budget')

    def test_rlvr_structured_contract_and_legacy_default(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = {'base': temp}
            self.assertEqual(rlvr_prompt_config({}, spec)['prompt_format'], 'vanilla')
            Path(temp, 'prompt_format.json').write_text('{"prompt_format":"structured"}')
            with self.assertRaises(ValueError): rlvr_prompt_config({}, spec)
            spec['prompt_format'] = 'structured'
            runtime = rlvr_prompt_config({}, spec)
            text = prompt('generation', Grid(2, 3, (), (0, 0), (0, 2)), input_format=runtime['prompt_format'])
            self.assertIn('Grid: 2x3', text)
            with self.assertRaises(ValueError): rlvr_prompt_config({'prompt_format':'vanilla'}, spec)

    def test_prefix_lock_rejects_test_evidence_and_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            report, lock = Path(temp, 'dev.json'), Path(temp, 'lock.json')
            identity = {'source_sha256': {'a': 'original'}, 'model_files': {'weights': 'original'}}
            record = {'split': 'test_id', 'identity': identity, 'metrics': {'success': 0}}
            report.write_text(json.dumps(record))
            with self.assertRaises(ValueError): create_lock(identity, report)
            record['split'] = 'dev'
            report.write_text(json.dumps(record))
            write_new_json(lock, create_lock(identity, report))
            self.assertEqual(validate_lock(lock, identity)['test_status'], 'reused_exploratory')
            with self.assertRaises(ValueError): validate_lock(lock, {'source_sha256': 'changed'})
            report.write_text('{}')
            with self.assertRaises(ValueError): validate_lock(lock, identity)

    def test_test_requires_lock_even_for_symbolic_control(self):
        for split in ('test_id', 'test_ood'):
            with self.assertRaisesRegex(ValueError, 'requires --freeze'):
                run({'symbolic_only': True}, split, 'unused')

    def test_selection_file_is_mandatory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'config.json').write_text(json.dumps({'output_root': str(root)}))
            (root / 'manifest.json').write_text(json.dumps({'executor_path': 'fixture', 'executor_sha256': 'hash'}))
            spec = {'family': 'full_finetune', 'config': str(root/'config.json'),
                    'dataset_root': str(root), 'traces': dict.fromkeys(('dev','test_id','test_ood'), str(root/'config.json'))}
            with patch('ppnl.prefix_evaluation.audit_splits', return_value={}), patch('ppnl.prefix_evaluation.OfficialOracle') as oracle:
                oracle.return_value.sha256 = 'hash'
                with self.assertRaises(FileNotFoundError): prepare(spec)

    def test_symbolic_dev_freeze_test_round_trip(self):
        from ppnl.data import write_jsonl
        grid = Grid(2, 3, (), (0, 0), (0, 2))
        class Oracle:
            check = staticmethod(execute)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace = root / 'trace.jsonl'
            row = {'grid': grid.__dict__, 'initial_candidate': 'up',
                   'modes': {name: {'candidate': 'up', 'truth': execute(grid, 'up').label(), 'model_calls': 1}
                             for name in ('direct', 'executor_one', 'executor_three')}}
            write_jsonl(trace, [row])
            for split in ('dev', 'test_id', 'test_ood'):
                write_jsonl(root / (split + '.jsonl'), [{'task': 'generation', 'grid': grid.__dict__}])
            spec = {'family': 'lora_gvr', 'symbolic_only': True, 'dataset_root': str(root),
                    'max_actions': 3, 'traces': dict.fromkeys(('dev','test_id','test_ood'), str(trace))}
            identity = {'fixture': 'immutable'}
            with patch('ppnl.prefix_evaluation.prepare', return_value=({}, Oracle(), root, identity)):
                dev = run(spec, 'dev', root/'dev')
                self.assertEqual(dev['metrics']['prefix_visit_heuristic']['success_rate'], 1)
                lock = root/'freeze.json'
                write_new_json(lock, create_lock(identity, root/'dev/results.json'))
                result = run(spec, 'test_id', root/'test', lock)
                self.assertEqual(result['n'], 1)
                self.assertIsNotNone(result['freeze_receipt'])
                with self.assertRaises(FileExistsError): run(spec, 'test_id', root/'test', lock)
