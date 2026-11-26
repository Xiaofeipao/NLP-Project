"""Data-only checks for Hope joint G/C/V reward experiments; no model dependencies."""
import ast
from collections import Counter, deque
import hashlib
import json
from pathlib import Path

CODE_COMMIT = '9ebecebe54e94e3bd474a5522c23df86ad66852d'
DATA_COMMIT = 'd88f85b872560f9313ac4319cdd5fa86151c0780'
MOVES = ('up', 'down', 'left', 'right')
DELTAS = dict(zip(MOVES, ((-1, 0), (1, 0), (0, -1), (0, 1))))
LABELS = ('valid', 'obstacle_collision', 'out_of_bounds', 'invalid_action_token', 'goal_not_reached')
TRAIN_SHA = '1a582bb9ac5ed58ab503f9f8ad44b0fe0e0f6d7f6a2dac9ea1d21c76745e9ef2'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


def fields(text):
    lines = dict(s.split(': ', 1) for s in text.splitlines() if ': ' in s)
    return (tuple(ast.literal_eval(lines['Current'])), tuple(ast.literal_eval(lines['Goal'])),
            {tuple(p) for p in ast.literal_eval(lines['Walls'])})


def category(current, goal, walls, action):
    if action not in DELTAS:
        return 'invalid_action_token', 'no_coordinate'
    nxt = tuple(a + b for a, b in zip(current, DELTAS[action]))
    cell = f'({nxt[0]},{nxt[1]})'
    if any(v < 0 or v > 5 for v in nxt):
        return 'out_of_bounds', cell
    if nxt in walls:
        return 'obstacle_collision', cell
    return ('valid' if nxt == goal else 'goal_not_reached'), cell


def distances(goal, walls):
    dist, queue = {goal: 0}, deque([goal])
    while queue:
        point = queue.popleft()
        for delta in DELTAS.values():
            nxt = tuple(a + b for a, b in zip(point, delta))
            if all(0 <= v < 6 for v in nxt) and nxt not in walls and nxt not in dist:
                dist[nxt] = dist[point] + 1
                queue.append(nxt)
    return dist


def reward_tables(corpus):
    generators = {r['case']: r for r in corpus if r['task'] == 'G'}
    coordinates = {r['id'][:-2]: r for r in corpus if r['task'] == 'C'}
    seen, rewards, kinds, count = {}, {}, {}, Counter()
    for r in corpus:
        if r['task'] != 'V':
            continue
        c = coordinates[r['id']]
        action = c['text'].splitlines()[1].removeprefix('Action: ')
        g = generators[r['case']]
        current, goal, walls = fields(g['text'])
        label, cell = category(current, goal, walls, action)
        assert c['answers'] == [cell] and r['answers'] == [label]
        assert r['text'].splitlines()[0] == 'Current: ' + cell
        if action not in MOVES:
            continue
        assert action not in seen.setdefault(r['case'], {})
        seen[r['case']][action] = label
    for case, g in generators.items():
        assert set(seen[case]) == set(MOVES)
        current, goal, walls = fields(g['text'])
        dist = distances(goal, walls)
        assert current in dist and current != goal
        optimal = set()
        vector, types = [], []
        for action in MOVES:
            label = seen[case][action]
            if label in ('out_of_bounds', 'obstacle_collision'):
                value, kind = -5.0, label
            else:
                nxt = tuple(a + b for a, b in zip(current, DELTAS[action]))
                is_optimal = dist[nxt] == dist[current] - 1
                value, kind = (1.0, 'optimal') if is_optimal else (-1.0, 'legal_detour')
                if is_optimal:
                    optimal.add(action)
            vector.append(value)
            types.append(kind)
            count[kind] += 1
        assert optimal == set(g['answers'])
        rewards[g['id']], kinds[g['id']] = vector, types
    assert len(generators) == 4000 and sum(count.values()) == 16000
    return rewards, kinds, dict(count)


def prepare(repo, dataset, out):
    repo, dataset, out = map(Path, (repo, dataset, out))
    source = repo / 'data/coordinate_gcv_review/train.jsonl'
    assert sha(source) == TRAIN_SHA
    corpus = rows(source)
    assert Counter(r['task'] for r in corpus) == {'G': 4000, 'C': 16400, 'V': 16400}
    rewards, kinds, counts = reward_tables(corpus)
    original = read(repo / 'artifacts/part1_baseline/data/protocol.json')
    sources = {'dev': repo / 'data/derived/dev.jsonl',
               'double_error': dataset / 'double_error_200.jsonl',
               'single_error': dataset / 'single_error_200.jsonl'}
    train_layouts = {tuple(sorted(fields(r['text'])[2])) for r in corpus if r['task'] == 'G'}
    heldout_layouts = set()
    for split, path in sources.items():
        gold = [r for r in rows(path) if r.get('task', 'verification') == 'verification']
        assert len(gold) == (100 if split == 'dev' else 200)
        layouts = {tuple(sorted(tuple(p) for p in r['grid']['obstacles'])) for r in gold}
        assert not layouts & train_layouts and not layouts & heldout_layouts
        heldout_layouts.update(layouts)
        write_rows(out / 'data' / (split + '.jsonl'),
                   [{k: r[k] for k in ('id', 'grid', 'candidate')} for r in gold])
        original['inputs'][split] = sha(out / 'data' / (split + '.jsonl'))
        original['source_splits'][split] = sha(path)
    original['experiment'] = {'code_commit': CODE_COMMIT, 'dataset_commit': DATA_COMMIT,
                              'dataset_source': str(dataset), 'shared_model': True,
                              'initialization': 'original pretrained Flan-T5-base, never a Hope fine-tuned checkpoint',
                              'reward': {'optimal': 1, 'legal_detour': -1, 'obstacle_collision': -5, 'out_of_bounds': -5},
                              'arms': ['supervised', 'expected', 'rloo'], 'seed': 42,
                              'reward_strength': 0.2, 'rloo_samples': 8, 'updates': 375,
                              'microbatch': 16, 'effective_batch_states': 32,
                              'checkpoint_policy': 'fixed final update, no test-based selection',
                              'training_sha256': TRAIN_SHA}
    save(out / 'data/protocol.json', original)
    summary = {'states': 4000, 'rows': len(corpus), 'four_action_categories': counts,
               'train_layouts': len(train_layouts), 'all_coordinate_and_verdict_labels_checked': True,
               'all_optimal_sets_checked_with_bfs': True, 'dataset_commit': DATA_COMMIT,
               'source_hashes': original['source_splits']}
    save(out / 'data_audit.json', summary)
    return corpus, rewards, kinds, sources
