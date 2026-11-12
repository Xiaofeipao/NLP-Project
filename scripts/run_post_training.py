"""Freeze and run inference ablations on an existing full G/V/R checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ppnl.full_finetune_gvr import FullFinetuneGVRModel
from ppnl.experiment_freeze import file_hash
from ppnl.harness import _lock, fingerprint
from ppnl.modeling import Seq2Seq, gpu_budget, require_cuda
from ppnl.post_training import (PostTrainingStudy, _read_record, _save_record, build_models, methods_from_config,
                                physical_cost_summary, preflight, prepare_output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'run'))
    parser.add_argument('--spec', default='configs/post_training.json')
    parser.add_argument('--out', default='outputs/post_training_reuse')
    parser.add_argument('--splits', nargs='+', choices=('dev', 'test_id', 'test_ood'))
    parser.add_argument('--limit', type=int, help='Smoke cases per task subset; a different frozen protocol')
    parser.add_argument('--resume', action='store_true')
    for option in ('checkpoint', 'model-config', 'dataset-root', 'executor', 'ledger'):
        parser.add_argument('--' + option)
    args = parser.parse_args(argv)
    spec = json.loads((ROOT / args.spec).read_text(encoding='utf-8'))
    for field in ('checkpoint', 'model_config', 'dataset_root', 'executor', 'ledger', 'splits', 'limit'):
        if getattr(args, field) is not None:
            spec[field] = getattr(args, field)
    config, oracle, rows_by_split, identity = preflight(spec, ROOT)
    methods = methods_from_config(spec)
    protocol = fingerprint(identity)
    print(json.dumps({'preflight': 'passed', 'protocol_sha256': protocol,
                      'methods': [method.name for method in methods],
                      'splits': {name: len(rows) for name, rows in rows_by_split.items()},
                      'smoke_limit': spec.get('limit')}, ensure_ascii=False), flush=True)
    if args.action == 'preflight':
        return
    output = ROOT / args.out
    prepare_output(output, identity, args.resume)
    # One process owns the whole batch; each reused GVR session also has its own lock.
    with _lock(output):
        prepare_output(output, identity, resume=True)
        torch = require_cuda()
        torch.set_num_threads(1)
        torch.manual_seed(spec['seed'])
        with gpu_budget(ROOT / spec['ledger'], spec['gpu_hours_cap'], 'post_training:' + protocol[:12]) as check:
            runner = Seq2Seq(str(ROOT / spec['checkpoint']), config)
            study = PostTrainingStudy(output, spec, protocol, oracle, FullFinetuneGVRModel(runner),
                                      build_models(runner, spec, methods), check)
            reports = {split: _read_record(output / split / 'results.json') for split in spec['splits']
                       if (output / split / 'results.json').exists()}
            for split, rows in rows_by_split.items():
                if split.startswith('test_'):
                    dev = output / 'dev/results.json'
                    _save_record(output / f'{split}_receipt.json', {
                        'protocol_sha256': protocol,
                        'development_report_sha256': file_hash(dev) if dev.exists() else None,
                        'method_selection': 'none; entire matrix frozen before any evaluation',
                        'test_status': 'smoke' if spec.get('limit') else 'reused_exploratory'})
                reports[split] = study.evaluate_split(rows, split)
                result = {'protocol_sha256': protocol, 'splits': reports,
                          'physical_cost': physical_cost_summary(output, spec['splits']),
                          'gpu': {'name': torch.cuda.get_device_name(),
                                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                  'torch': torch.__version__, 'cuda': torch.version.cuda,
                                  'python': platform.python_version()}}
                _save_record(output / 'results.json', result)
                print(json.dumps({'completed': split, 'subsets': reports[split]['subset_counts'],
                                  'physical_model_calls': result['physical_cost']['total']['model_calls']}), flush=True)
    print(json.dumps({'output': str(output), 'status': 'completed'}), flush=True)


if __name__ == '__main__':
    main()
