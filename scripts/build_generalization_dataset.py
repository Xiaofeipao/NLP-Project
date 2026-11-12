"""Build separate, reproducible synthetic grid-shift evaluation sets; no GPU."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ppnl.core import Grid, OfficialOracle, shortest_path
from ppnl.data import make_candidate, read_jsonl, write_json, write_jsonl
from ppnl.gvr_evaluation import evaluation_subsets

KINDS = ('substitution', 'deletion', 'insertion', 'premature_stopping', 'invalid_token')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output, seed=20261112):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new output directory; never overwrite frozen evaluation data')
    # Fixed source roots keep regeneration independent of other output copies.
    source_roots = [ROOT / 'data' / name for name in ('derived', 'policy_v2', 'policy_v3')]
    sources = sorted(p for directory in source_roots for p in directory.glob('*.jsonl'))
    sources += [directory / 'manifest.json' for directory in source_roots]
    source_hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    reserved = set()
    for path in sources:
        if path.suffix == '.jsonl':
            reserved.update(row['layout_id'] for row in read_jsonl(path))
        else:
            manifest = json.loads(path.read_text(encoding='utf-8'))
            for layouts in manifest.get('layout_splits', {}).values():
                reserved.update(layouts)
    # Include all official layout partitions, not just sampled training rows.
    train = read_jsonl(ROOT / 'data/derived/train.jsonl')
    obstacle_counts = [len(row['grid']['obstacles']) for row in train]
    oracle = OfficialOracle(ROOT / 'vendor/ppnl/executor-point-sg.py')
    rng = random.Random(seed)
    used = set(reserved)
    datasets, summaries = {}, {}
    for split, size in [('test_7x7', 7), ('test_dense_6x6', 6)]:
        rows = []
        rejected = Counter()
        for attempt in range(100000):
            if len(rows) == 200:
                break
            # Approximately preserve the empirical train obstacle density for 7x7.
            count = round(rng.choice(obstacle_counts) * 49 / 36) if size == 7 else rng.randint(9, 12)
            cells = [(r, c) for r in range(size) for c in range(size)]
            obstacles = tuple(sorted(rng.sample(cells, count)))
            start, goal = rng.sample([p for p in cells if p not in obstacles], 2)
            grid = Grid(size, size, obstacles, start, goal)
            if grid.layout_id in used:
                rejected['duplicate_or_reserved_layout'] += 1
                continue
            base = shortest_path(grid)
            if base is None or len(base) < 2:
                rejected['unreachable_or_path_shorter_than_two'] += 1
                continue
            i = len(rows)
            valid = i % 2 == 0
            long_valid = valid and i % 4 == 0
            candidate, verdict, edits = make_candidate(
                grid, base, rng, oracle, valid, 1, long_valid, KINDS[(i // 2) % 5])
            assert oracle.check(grid, ' '.join(base)).valid
            rows.append({
                'id': f'{split}-{i}', 'split': split, 'layout_id': grid.layout_id,
                'grid': asdict(grid), 'source_id': f'synthetic:{seed}:{split}:{attempt}',
                'candidate': candidate, 'label': verdict.label(), 'valid': verdict.valid,
                'reason': verdict.reason, 'step': verdict.step, 'trace': verdict.trace,
                'target_plan': ' '.join(base), 'edits': edits, 'edit_count': len(edits),
                'valid_kind': ('nonoptimal' if long_valid else 'optimal') if valid else None,
            })
            used.add(grid.layout_id)
        if len(rows) != 200:
            raise RuntimeError(f'Could not fill {split}')
        verify, repair, generate = evaluation_subsets(rows)
        assert (len(verify), len(repair), len(generate)) == (200, 100, 200)
        assert Counter(r['valid_kind'] for r in rows) == {None: 100, 'optimal': 50, 'nonoptimal': 50}
        assert Counter(r['edits'][0]['kind'] for r in repair) == dict.fromkeys(KINDS, 20)
        datasets[split] = rows
        lengths = [len(r['target_plan'].split()) for r in rows]
        summaries[split] = {
            'environments': 200, 'unique_layouts': len({r['layout_id'] for r in rows}),
            'generation': 200, 'verification': 200, 'invalid_candidate_repair': 100,
            'valid_shortest': 50, 'valid_detour': 50, 'invalid_single_edit': 100,
            'obstacle_counts': dict(sorted(Counter(len(r['grid']['obstacles']) for r in rows).items())),
            'shortest_path_length': {'min': min(lengths), 'max': max(lengths), 'mean': sum(lengths)/200},
            'rejected': dict(rejected),
        }
    # Ensure the original datasets were not changed while constructing the extension.
    assert source_hashes == {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    output.mkdir(parents=True)
    for split, rows in datasets.items():
        write_jsonl(output / f'{split}.jsonl', rows)
    write_json(output / 'manifest.json', {
        'schema_version': 1, 'seed': seed, 'purpose': 'optional_evaluation_only',
        'provenance': 'project-generated synthetic grids; not official downloaded samples',
        'model_evaluation_status': 'not_run', 'counts': summaries,
        'layout_splits': {s: sorted(r['layout_id'] for r in rows) for s, rows in datasets.items()},
        'source_sha256': source_hashes,
        'generator_sha256': digest(Path(__file__)),
        'semantics_sha256': {p: digest(ROOT / p) for p in ['ppnl/core.py', 'ppnl/data.py']},
        'executor_sha256': oracle.sha256,
        'files_sha256': {f'{s}.jsonl': digest(output / f'{s}.jsonl') for s in datasets},
        'rules': {
            '7x7_density': 'sample train row obstacle count; round(count * 49 / 36)',
            'dense_6x6_density': 'uniform integer 9..12 obstacles (25%..33.33% of all cells)',
            'sampling': 'uniform obstacle cells, then uniform distinct free start/goal; reject unreachable or shortest length < 2',
            'leakage': 'exclude all known manifest partitions and existing dataset layouts; one environment per new layout',
            'training': 'never mix into train/dev; original 4000/200 quotas unchanged',
            'evaluation': 'freeze model, code and decoding budgets separately before model evaluation; do not select on these sets',
            'interpretation': 'synthetic grid-distribution shift; density matching is approximate, not a pure isolated size intervention',
            'no_path': 'all included environments reachable; unreachable recognition is a separate experiment',
            'budget': 'do not silently reuse the historical 36-action cap for 7x7; predeclare fixed or size-scaled budgets',
        },
    })
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/generalization_v1')
    parser.add_argument('--seed', type=int, default=20261112)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.seed), indent=2))
