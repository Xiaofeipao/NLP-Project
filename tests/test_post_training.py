from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from ppnl.constrained_decoding import DecodeSpec
from ppnl.core import Grid, execute
from ppnl.harness import fingerprint
from ppnl.post_training import (Method, PostTrainingStudy, RuleModel, _read_record,
                                physical_cost_summary, prepare_output, validate_spec)


class LocalOracle:
    def check(self, grid, plan):
        return execute(grid, plan)


class Model:
    def __init__(self, fail_once=False):
        self.calls = []
        self.last_usage = {}
        self.fail_once = fail_once

    def predict(self, task, grid, candidate='', feedback='', check=lambda: None):
        check()
        self.calls.append((task, candidate, feedback))
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError('Interrupted backend')
        self.last_usage = {'input_tokens': 3, 'output_tokens': 1, 'repair_feedback_used': feedback}
        # Intentionally wrong V: the workflow must not silently replace it with gold.
        return {'generation': 'left', 'verification': 'valid', 'repair': 'right'}[task]


class PostTrainingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.grid = Grid(2, 2, (), (0, 0), (0, 1))
        self.spec = {'methods': [{'name': 'direct', 'mode': 'direct'},
                                 {'name': 'self', 'mode': 'self'}, {'name': 'executor_one'},
                                 {'name': 'uniform', 'rule': 'uniform', 'decoding': 'constrained'}],
                     'model_calls': 8, 'case_seconds': 30, 'beam_width': 4,
                     'max_actions': 8, 'max_score_rows': 30, 'splits': ['test_id', 'test_ood'],
                     'seed': 42, 'gpu_hours_cap': 20}
        self.base, self.self_model, self.executor = Model(), Model(), Model()
        self.models = {'direct': Model(), 'self': self.self_model, 'executor_one': self.executor,
                       'uniform': RuleModel('uniform', DecodeSpec(max_actions=8, max_score_rows=30))}
        self.study = PostTrainingStudy(self.output, self.spec, 'protocol', LocalOracle(), self.base, self.models)

    def rows(self, split):
        result = []
        for index, candidate in enumerate(('right', 'left')):
            truth = execute(self.grid, candidate)
            result.append({'id': f'{split}-{index}', 'grid': asdict(self.grid), 'candidate': candidate,
                           'valid': truth.valid, 'label': truth.label(), 'step': truth.step, 'reason': truth.reason})
        return result

    def test_model_feedback_controls_self_but_executor_uses_gold(self):
        report = self.study.evaluate_split(self.rows('test_id'), 'test_id')
        self.assertEqual(report['candidate_repair']['self']['repair_success_rate'], 0)
        self.assertEqual(report['candidate_repair']['self']['false_valid_accepts'], 1)
        self.assertEqual(report['candidate_repair']['executor_one']['repair_success_rate'], 1)
        self.assertEqual(report['generation']['self']['success_rate'], 0)
        self.assertTrue(all(task == 'verification' for task, _, _ in self.self_model.calls))
        record = _read_record(self.output / 'test_id/candidate_traces.json')['rows'][0]
        self.assertEqual(record['model_feedback'], 'valid')
        self.assertEqual(record['gold_feedback'], execute(self.grid, 'left').label())
        self.assertEqual(record['methods']['executor_one']['history'][-1]['repair_feedback_used'], record['gold_feedback'])
        self.assertIn('first_error_localization_accuracy', report['verification'])
        self.assertIn('out_of_bounds', report['repair_by_error'])

    def test_pairing_resume_and_costs_count_only_actual_model_requests(self):
        first = self.study.evaluate_split(self.rows('test_id'), 'test_id')
        second = self.study.evaluate_split(self.rows('test_ood'), 'test_ood')
        self.assertEqual(first['generation'], second['generation'])
        self.assertEqual(sum(task == 'generation' for task, _, _ in self.base.calls), 1)
        calls = (len(self.base.calls), len(self.self_model.calls), len(self.executor.calls))
        self.study.evaluate_split(self.rows('test_id'), 'test_id')
        self.assertEqual(calls, (len(self.base.calls), len(self.self_model.calls), len(self.executor.calls)))
        cost = physical_cost_summary(self.output, self.spec['splits'])
        self.assertEqual(cost['unique_generation_environments'], 1)
        self.assertEqual(cost['total']['model_calls'], sum(calls))
        rule = first['candidate_repair']['uniform']['cost']
        self.assertEqual(rule['model_calls_after_initial'], 0)
        self.assertEqual(rule['input_tokens'], 0)
        self.assertEqual(rule['decoder_row_steps'], 0)
        self.assertGreater(rule['rule_search_rows'], 0)

    def test_verification_failure_is_charged_and_cache_input_is_bound(self):
        self.base.fail_once = True
        row = self.rows('test_id')[1]
        path = self.output / 'verification.json'
        with self.assertRaisesRegex(RuntimeError, 'Interrupted'):
            self.study._verify(row, path)
        self.assertEqual(_read_record(path)['model_calls'], 1)
        record = self.study._verify(row, path)
        self.assertEqual(record['model_calls'], 2)
        with self.assertRaisesRegex(ValueError, 'another study/input'):
            self.study._verify({**row, 'id': 'another-row'}, path)

    def test_shared_generation_rejects_transplanted_environment_cache(self):
        self.study._initial(self.grid)
        other = Grid(2, 2, (), (0, 0), (1, 1))
        parent = self.output / 'shared_generation'
        shutil.copytree(parent / fingerprint(asdict(self.grid)), parent / fingerprint(asdict(other)))
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.study._initial(other)

    def test_resume_requires_exact_protocol_and_integrity(self):
        output = self.output / 'frozen'
        identity = {'checkpoint': 'sha', 'methods': self.spec['methods']}
        digest = prepare_output(output, identity)
        self.assertEqual(prepare_output(output, identity, resume=True), digest)
        with self.assertRaises(FileExistsError):
            prepare_output(output, identity)
        with self.assertRaisesRegex(ValueError, 'protocol changed'):
            prepare_output(output, {**identity, 'checkpoint': 'changed'}, resume=True)
        path = output / 'protocol.json'
        value = json.loads(path.read_text())
        value['checkpoint'] = 'tampered'
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'Corrupt'):
            prepare_output(output, identity, resume=True)

    def test_protocol_rejects_silent_ablation_changes(self):
        validate_spec(self.spec)
        for change in ({'limit': 0}, {'limit': True}, {'splits': []}, {'splits': ['test_id', 'test_id']},
                       {'model_calls': True}, {'case_seconds': float('nan')}, {'gpu_hours_cap': 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_spec({**self.spec, **change})
        for kwargs in ({'mode': 'self', 'decoding': 'constrained'}, {'mode': 'direct', 'feedback': 'none'},
                       {'rule': 'uniform'}, {'name': '../outside'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Method(**{'name': 'method', **kwargs})

    def test_cli_help_does_not_load_gpu(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(root / 'scripts/run_post_training.py'), '--help'],
                                capture_output=True, text=True, check=True)
        self.assertIn('--resume', result.stdout)
        self.assertIn('--checkpoint', result.stdout)


if __name__ == '__main__':
    unittest.main()
