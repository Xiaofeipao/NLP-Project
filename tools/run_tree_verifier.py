"""Train a CPU decision-tree verifier using the existing frozen V subset."""
import json
from pathlib import Path
import sys
import hashlib
import joblib
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ppnl.core import ACTIONS, Grid, execute
from ppnl.data import read_jsonl
from ppnl.evaluation import verification_metrics
from ppnl.experiment_freeze import audit_splits


def features(grid, position, action):
    dr, dc = ACTIONS.get(action, (0, 0))
    return [*position, *grid.goal, position[0]+dr, position[1]+dc,
            position[0]-grid.goal[0], position[1]-grid.goal[1],
            *[int(action == token) for token in (*ACTIONS, '<eos>', '<unknown>')],
            *[int((r,c) in grid.obstacles) for r in range(6) for c in range(6)]]


def events(grid, candidate):
    position = grid.start
    tokens = candidate.split()
    for index, token in enumerate([*tokens, '<eos>'], 1):
        action = '<eos>' if index == len(tokens)+1 else token if token in ACTIONS else '<unknown>'
        yield index, features(grid, position, action)
        dr, dc = ACTIONS.get(token, (0, 0))
        position = position[0]+dr, position[1]+dc


def predict(tree, grid, candidate):
    for step, vector in events(grid, candidate):
        reason = tree.predict([vector])[0]
        if reason != 'ok':
            return f'invalid | step={step} | reason={reason}'
    return 'valid'


def main():
    out = ROOT/'outputs/tree_verifier_001'
    if out.exists():
        raise FileExistsError(out)
    audit_splits('data/derived')
    rows = {split: read_jsonl(ROOT/'data/derived'/f'{split}.jsonl')
            for split in ('train','dev','test_id','test_ood')}
    for split in ('train','dev'):
        rows[split] = [r for r in rows[split] if r['task']=='verification']
    x, y = [], []
    for row in rows['train']:
        grid = Grid.from_dict(row['grid'])
        truth = execute(grid, row['candidate'])
        for step, vector in events(grid,row['candidate']):
            x.append(vector)
            y.append(truth.reason if step == truth.step else 'ok')
            if step == truth.step:
                break
    candidates = []
    for depth in (4,8,12,None):
        tree = DecisionTreeClassifier(max_depth=depth,min_samples_leaf=2,random_state=42)
        tree.fit(x,y)
        predictions = [predict(tree,Grid.from_dict(r['grid']),r['candidate']) for r in rows['dev']]
        metrics = verification_metrics(rows['dev'],predictions)
        candidates.append((tree,metrics,depth))
    tree, dev, depth = max(candidates,key=lambda c:(c[1]['macro_f1'],c[1]['joint_error_accuracy']))
    out.mkdir(parents=True)
    joblib.dump(tree,out/'tree.joblib')
    report = {'training_candidates':len(rows['train']),'expanded_steps':len(x),'selected_depth':depth,
              'selection':'dev macro-F1 then joint error accuracy; ties prefer shallower tree',
              'dev_candidates':[{'depth':d,'metrics':m} for _,m,d in candidates],
              'test_status':'reused_exploratory','metrics':{'dev':dev},
              'training_classes':sorted(set(y))}
    (out/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    report['selected_model_sha256']=hashlib.sha256((out/'tree.joblib').read_bytes()).hexdigest()
    for split in ('test_id','test_ood'):
        predictions = [predict(tree,Grid.from_dict(r['grid']),r['candidate']) for r in rows[split]]
        report['metrics'][split] = verification_metrics(rows[split],predictions)
        (out/f'{split}_predictions.json').write_text(json.dumps([
            {'id':r['id'],'prediction':p,'truth':r['label']} for r,p in zip(rows[split],predictions)],indent=2),encoding='utf-8')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'tree.txt').write_text(export_text(tree,max_depth=5),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
