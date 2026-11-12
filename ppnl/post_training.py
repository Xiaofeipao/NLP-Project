"""Paired inference ablations on one full multitask checkpoint.

The existing GVR state machine owns decisions and durable model requests.
This module owns the method matrix, shared initial plans and batch summaries.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import time

from .constrained_decoding import DecodeSpec
from .core import Grid, OfficialOracle, execute
from .data import read_jsonl
from .evaluation import plan_metrics, verification_metrics
from .experiment_freeze import audit_splits, file_hash
from .full_finetune_gvr import FullFinetuneGVRModel, checkpoint_identity
from .gvr_evaluation import evaluation_subsets
from .gvr_harness import (USAGE_COUNTERS, create_gvr_session, run_gvr_session,
                          score_gvr_session, validate_gvr_protocol)
from .gvr_rule_controls import rule_repair
from .harness import _atomic_json, fingerprint


@dataclass(frozen=True)
class Method:
    name: str
    mode: str = 'executor_one'
    decoding: str = 'ordinary'
    feedback: str = 'original'
    rule: str | None = None

    def __post_init__(self):
        if not re.fullmatch(r'[a-z][a-z0-9_]*', self.name):
            raise ValueError('Method names must be safe lowercase identifiers')
        if self.mode not in ('direct', 'self', 'executor_one', 'executor_three'):
            raise ValueError('Unknown GVR mode')
        if self.decoding not in ('ordinary', 'syntax_beam', 'constrained', 'backtrack'):
            raise ValueError('Unknown repair decoding')
        if self.feedback not in ('original', 'none', 'generic'):
            raise ValueError('Unknown repair feedback')
        if self.rule not in (None, 'uniform', 'manhattan'):
            raise ValueError('Unknown rule control')
        if self.mode in ('direct', 'self') and (
                self.decoding != 'ordinary' or self.feedback != 'original' or self.rule):
            raise ValueError('Direct/self require ordinary model-only task decoding')
        if self.rule and (self.decoding != 'constrained' or self.feedback != 'original'):
            raise ValueError('Rule controls require constrained decoding and original feedback')


def methods_from_config(spec):
    methods = [Method(**item) for item in spec['methods']]
    if not methods or len({method.name for method in methods}) != len(methods):
        raise ValueError('Provide a nonempty matrix with unique method names')
    return methods


def validate_spec(spec):
    methods = methods_from_config(spec)
    for method in methods:
        validate_gvr_protocol(method.mode, spec['model_calls'], spec['case_seconds'])
        decoding_spec(method, spec)
    splits = spec['splits']
    if (not isinstance(splits, list) or not splits or len(set(splits)) != len(splits)
            or any(split not in ('dev', 'test_id', 'test_ood') for split in splits)):
        raise ValueError('Select unique dev/test_id/test_ood splits')
    if 'limit' in spec and (type(spec['limit']) is not int or spec['limit'] < 1):
        raise ValueError('Smoke limit must be a positive integer; omit for a full run')
    cap = spec['gpu_hours_cap']
    if (isinstance(cap, bool) or not isinstance(cap, (float, int))
            or not math.isfinite(cap) or cap <= 0 or type(spec['seed']) is not int):
        raise ValueError('Invalid GPU-hour cap or seed')
    return methods


def decoding_spec(method, spec):
    if method.decoding == 'ordinary':
        return None
    return DecodeSpec(beam_width=spec['beam_width'], max_actions=spec['max_actions'],
                      max_score_rows=spec['max_score_rows'],
                      state_constraints=method.decoding != 'syntax_beam',
                      backtrack=method.decoding == 'backtrack')


def _read_record(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    digest = value.pop('record_sha256', None)
    if digest != fingerprint(value):
        raise ValueError(f'Corrupt cached result: {path}')
    return value


def _save_record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, {**value, 'record_sha256': fingerprint(value)})


class InitialPlanModel:
    """Expose a stored initial candidate while forwarding V/R to one method."""
    def __init__(self, model, initial):
        self.model, self.initial = model, initial
        self.last_usage = {}
        self.repair_decoding = getattr(model, 'repair_decoding', None)

    def predict(self, task, grid, candidate='', feedback='', check=lambda: None):
        self.last_usage = {}
        if task == 'generation':
            self.last_usage = {'input_tokens': 0, 'output_tokens': 0}
            return self.initial
        result = self.model.predict(task, grid, candidate, feedback, check)
        self.last_usage = dict(self.model.last_usage)
        return result


class RuleModel:
    def __init__(self, kind, spec):
        self.kind, self.repair_decoding = kind, spec
        self.last_usage = {}

    def predict(self, task, grid, candidate='', feedback='', check=lambda: None):
        if task != 'repair':
            raise ValueError('Rule control only implements repair')
        result = rule_repair(grid, self.kind, self.repair_decoding, check)
        self.last_usage = {'decode_status': result.stop_reason,
                           'decoder_row_steps': result.decoder_row_steps,
                           'input_tokens': 0, 'output_tokens': 0,
                           'repair_feedback_used': None}
        return result.plan


def build_models(runner, spec, methods):
    models = {}
    for method in methods:
        decoding = decoding_spec(method, spec)
        models[method.name] = (RuleModel(method.rule, decoding) if method.rule else
                               FullFinetuneGVRModel(runner, decoding, method.feedback))
    return models


class PostTrainingStudy:
    def __init__(self, output, spec, identity, oracle, base, models, check=lambda: None):
        self.output, self.spec = Path(output), spec
        self.identity, self.oracle = identity, oracle
        self.base, self.models, self.check = base, models, check
        self.methods = methods_from_config(spec)

    def _session(self, path, grid, mode, model, identity):
        if not (path / 'manifest.json').exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            create_gvr_session(path, grid, mode, identity,
                               model_calls=self.spec['model_calls'],
                               active_seconds=self.spec['case_seconds'])
        state = run_gvr_session(path, model, identity, self.check)
        score = score_gvr_session(path, self.oracle)
        score['history'] = state['history']
        score['execution_trace'] = execute(grid, score['candidate']).trace
        return score

    def _initial(self, grid):
        key = fingerprint(asdict(grid))
        directory = self.output / 'shared_generation' / key
        score = self._session(directory, grid, 'direct', self.base,
                              {'study': self.identity, 'stage': 'initial_generation', 'grid': key})
        return score

    def _methods(self, directory, grid, initial):
        results = {}
        for method in self.methods:
            self.check()
            identity = {'study': self.identity, 'method': asdict(method),
                        'grid': fingerprint(asdict(grid)), 'initial': initial}
            score = self._session(directory / method.name, grid, method.mode,
                                  InitialPlanModel(self.models[method.name], initial), identity)
            score['controller_requests'] = score.pop('model_calls')
            score['model_calls_after_initial'] = (
                0 if method.rule else max(0, score['controller_requests'] - 1))
            score['rule_search_rows'] = score['decoder_row_steps'] if method.rule else 0
            if method.rule:
                score['decoder_row_steps'] = 0
            score['feedback_kind'] = 'rule' if method.rule else (
                'model' if method.mode == 'self' else 'none' if method.mode == 'direct'
                else method.feedback)
            results[method.name] = score
        return results

    def _verify(self, row, path):
        grid = Grid.from_dict(row['grid'])
        truth = self.oracle.check(grid, row['candidate']).label()
        if truth != row['label']:
            raise ValueError(f'Gold feedback mismatch: {row["id"]}')
        record = _read_record(path) if path.exists() else {
            'study': self.identity, 'input_sha256': fingerprint(row),
            'id': row['id'], 'grid': row['grid'], 'candidate': row['candidate'],
            'gold_feedback': truth, 'model_calls': 0, 'seconds': 0.0, 'completed': False}
        if record['study'] != self.identity or record['input_sha256'] != fingerprint(row):
            raise ValueError('Verification cache belongs to another study/input')
        if record['completed']:
            return record
        record['model_calls'] += 1
        _save_record(path, record)
        started = time.perf_counter()
        try:
            record['model_feedback'] = self.base.predict('verification', grid, row['candidate'],
                                                        check=self.check)
            record['usage'] = dict(self.base.last_usage)
            record['completed'] = True
        finally:
            record['seconds'] += time.perf_counter() - started
            _save_record(path, record)
        return record

    def evaluate_split(self, rows, split):
        verify, repair, generate = evaluation_subsets(rows)
        limit = self.spec.get('limit')
        if limit:
            verify, repair, generate = verify[:limit], repair[:limit], generate[:limit]
        out = self.output / split
        out.mkdir(parents=True, exist_ok=True)
        predictions = []
        verification_records = []
        for row in verify:
            self.check()
            record = self._verify(row, out / 'verification' / (fingerprint(row['id']) + '.json'))
            predictions.append(record['model_feedback'])
            verification_records.append(record)
        generation_records = []
        seen = set()
        for row in generate:
            grid = Grid.from_dict(row['grid'])
            key = fingerprint(asdict(grid))
            if key in seen:
                continue
            seen.add(key)
            initial = self._initial(grid)
            methods = self._methods(self.output / 'planning' / key, grid, initial['candidate'])
            generation_records.append({'id': row['id'], 'grid': row['grid'], 'environment_key': key,
                                       'initial': initial, 'methods': methods})
        repair_records = []
        for row in repair:
            grid = Grid.from_dict(row['grid'])
            truth = self.oracle.check(grid, row['candidate'])
            if truth.valid or truth.label() != row['label']:
                raise ValueError(f'Invalid repair gold label: {row["id"]}')
            methods = self._methods(out / 'repair' / fingerprint(row['id']), grid, row['candidate'])
            repair_records.append({'id': row['id'], 'grid': row['grid'], 'candidate': row['candidate'],
                                   'gold_feedback': truth.label(), 'reason': truth.reason,
                                   'model_feedback': next((item['model_feedback'] for item in verification_records
                                                           if item['id'] == row['id']), None),
                                   'methods': methods})
        verification = verification_metrics(verify, predictions)
        groups = defaultdict(list)
        for row, predicted in zip(verify, predictions):
            groups[row['reason'] or 'valid'].append((row, predicted))
        report = {'split': split, 'protocol_sha256': self.identity,
                  'test_status': 'smoke' if limit else 'reused_exploratory',
                  'subset_counts': {'verification': len(verify), 'repair': len(repair),
                                    'generation': len(generation_records)},
                  'verification': verification,
                  'verification_by_error': {reason: verification_metrics(
                      [row for row, _ in group], [value for _, value in group])
                      for reason, group in groups.items()},
                  'generation': self._summarize(generation_records),
                  'candidate_repair': self._summarize(repair_records, repair=True),
                  'generation_by_initial_error': {reason: self._summarize(
                      [record for record in generation_records
                       if (execute(Grid.from_dict(record['grid']), record['initial']['candidate']).reason or 'valid') == reason])
                      for reason in sorted({execute(Grid.from_dict(record['grid']), record['initial']['candidate']).reason or 'valid'
                                            for record in generation_records})},
                  'repair_by_error': {reason: self._summarize(
                      [record for record in repair_records if record['reason'] == reason], repair=True)
                      for reason in sorted({record['reason'] for record in repair_records})},
                  'definitions': {'candidate_self': 'model feedback gates repair; gold is for scoring only',
                      'candidate_executor': 'strict executor feedback, modified by the selected text ablation',
                      'generation': 'unique grid/start/goal; initial G and planning sessions shared across splits',
                      'cost': 'per-method cost excludes shared initial G; controller requests include cached G and rule requests',
                      'success_rate': 'goal reached without any error / all cases',
                      'repair_success_rate': 'success rate over originally invalid supplied candidates',
                      'feasibility': 'all actions legal and in bounds without obstacles; goal not required',
                      'optimality': 'successful shortest-length plans / all cases',
                      'rule_search_rows': 'symbolically scored prefixes, excluded from neural decoder_row_steps'}}
        _save_record(out / 'verification_traces.json', {'rows': verification_records})
        _save_record(out / 'generation_traces.json', {'rows': generation_records})
        _save_record(out / 'candidate_traces.json', {'rows': repair_records})
        cases = sorted(repair_records, key=lambda record: sum(
            value['success'] for value in record['methods'].values()))
        _save_record(out / 'case_studies.json', {'rows': cases[:max(2, min(10, len(cases)))]})
        _save_record(out / 'results.json', report)
        return report

    def _summarize(self, records, repair=False):
        result = {}
        for method in self.methods:
            values = [record['methods'][method.name] for record in records]
            metrics = plan_metrics([Grid.from_dict(record['grid']) for record in records],
                                   [value['candidate'] for value in values])
            metrics['false_valid_accepts'] = sum(value['false_valid_accept'] for value in values)
            if repair:
                metrics['repair_success_rate'] = metrics['success_rate']
            metrics['stop_reason_counts'] = dict(Counter(value['stop_reason'] for value in values))
            metrics['cost'] = {field: sum(value.get(field, 0) for value in values) for field in (
                'controller_requests', 'model_calls_after_initial', 'executor_calls', 'active_seconds',
                'executor_seconds', 'rule_search_rows', *USAGE_COUNTERS)}
            result[method.name] = metrics
        return result


def preflight(spec, root):
    root = Path(root)
    methods = validate_spec(spec)
    config = json.loads((root / spec['model_config']).read_text(encoding='utf-8'))
    config['local_files_only'] = True
    config['seed'] = spec['seed']
    dataset = root / spec['dataset_root']
    split_audit = audit_splits(dataset)
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    oracle = OfficialOracle(root / spec['executor'])
    if manifest['executor_sha256'] != oracle.sha256:
        raise ValueError('Official executor differs from the dataset manifest')
    rows_by_split = {}
    for split in spec['splits']:
        rows = read_jsonl(dataset / f'{split}.jsonl')
        counts = tuple(map(len, evaluation_subsets(rows)))
        expected = (100, 60, 40) if split == 'dev' else (200, 100, 200)
        if counts != expected:
            raise ValueError(f'{split} quotas are {counts}; expected {expected}')
        rows_by_split[split] = rows
    if 'test_id' in rows_by_split and 'test_ood' in rows_by_split:
        if [row['grid'] for row in rows_by_split['test_id']] != [row['grid'] for row in rows_by_split['test_ood']]:
            raise ValueError('ID/OOD environment pairing changed')
    checkpoint = root / spec['checkpoint']
    checkpoint_config = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    if any(method.decoding != 'ordinary' and not method.rule for method in methods):
        if checkpoint_config.get('model_type') != 't5':
            raise ValueError('Neural action decoding is currently validated only for T5 checkpoints')
    for path in checkpoint.rglob('*'):
        if path.is_file() and path.suffix in ('.bin', '.safetensors'):
            with path.open('rb') as handle:
                if handle.read(100).startswith(b'version https://git-lfs.github.com/spec/v1'):
                    raise ValueError(f'Model weights are still a Git LFS pointer: {path}')
    model = checkpoint_identity(checkpoint, config)
    sources = sorted((root / 'ppnl').glob('*.py')) + [root / 'scripts/run_post_training.py']
    identity = {'schema_version': 1, 'spec': spec, 'model': model, 'split_audit': split_audit,
                'design': 'preregistered full matrix; no development or test method selection',
                'source_sha256': {path.relative_to(root).as_posix(): file_hash(path) for path in sources},
                'data_sha256': {split: file_hash(dataset / f'{split}.jsonl') for split in spec['splits']},
                'manifest_sha256': file_hash(dataset / 'manifest.json'), 'executor_sha256': oracle.sha256}
    return config, oracle, rows_by_split, identity


def prepare_output(output, identity, resume=False):
    output = Path(output)
    manifest = output / 'protocol.json'
    if output.exists():
        if not resume:
            raise FileExistsError('Output exists; use --resume to continue its identical protocol')
        stored = _read_record(manifest)
        if stored != identity:
            raise ValueError('Checkpoint, data, source, runtime or method protocol changed')
    else:
        output.mkdir(parents=True)
        _save_record(manifest, identity)
    return fingerprint(identity)


def physical_cost_summary(output, splits):
    """Accrued costs of completed splits, counting paired G/planning only once."""
    generation, repair, verification = {}, [], []
    for split in splits:
        directory = Path(output) / split
        if not (directory / 'results.json').exists():
            continue
        for row in _read_record(directory / 'generation_traces.json')['rows']:
            generation[row['environment_key']] = row
        repair.extend(_read_record(directory / 'candidate_traces.json')['rows'])
        verification.extend(_read_record(directory / 'verification_traces.json')['rows'])

    def totals(values, calls='model_calls'):
        fields = (*USAGE_COUNTERS, 'executor_calls', 'executor_seconds', 'active_seconds', 'rule_search_rows')
        return {'model_calls': sum(value.get(calls, 0) for value in values),
                **{field: sum(value.get(field, 0) for value in values) for field in fields}}

    components = {
        'shared_initial_generation': totals([row['initial'] for row in generation.values()]),
        'planning_after_initial': totals([value for row in generation.values() for value in row['methods'].values()],
                                          'model_calls_after_initial'),
        'candidate_processing': totals([value for row in repair for value in row['methods'].values()],
                                         'model_calls_after_initial'),
        'independent_verification': totals([{**row.get('usage', {}), 'model_calls': row['model_calls'],
                                           'active_seconds': row['seconds']} for row in verification])}
    return {'components': components, 'total': {field: sum(value[field] for value in components.values())
                                                for field in components['shared_initial_generation']},
            'unique_generation_environments': len(generation),
            'definition': 'Cumulative actual requests for completed splits; paired environment sessions counted once. '
                          'Model loading and host idle time are in the GPU ledger, not task active_seconds. '
                          'Interrupted incomplete splits retain charged attempts in their session journals.'}
