"""GPU paired V/rule ablations with the previously selected R frozen."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from ppnl.core import Grid, execute, parse_verdict
from ppnl.data import read_jsonl, write_json, write_jsonl
from ppnl.evaluation import plan_metrics, verification_metrics
from ppnl.modeling import Seq2Seq, gpu_budget, require_cuda
from ppnl.policy import NeuralPolicy
from ppnl.prompts import prompt
from ppnl.repair_variants import repair
from ppnl.verifier_guard import guard_verdict
from part2.run_rollback import reported_prefix


def run_repair(grid, candidate, feedback, scorer, frozen):
    prefix, status = reported_prefix(grid, candidate, feedback, frozen['backoff'])
    if prefix is None:
        return candidate, status
    control = f'invalid | step={len(prefix.split())+1} | reason=goal_not_reached'
    output, status, _ = repair(grid, prefix, control, scorer, frozen['backend'])
    return output, status
from part2.run_rollback import reported_prefix


def run_repair(grid, candidate, feedback, scorer, frozen):
    # Predicted indices are never corrected using the true first error.
    prefix, status = reported_prefix(grid, candidate, feedback, frozen['backoff'])
    if prefix is None:
        return candidate, status
    control = f'invalid | step={len(prefix.split())+1} | reason=goal_not_reached'
    result, status, _ = repair(grid, prefix, control, scorer, frozen['backend'])
    return result, status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    require_cuda()
    frozen = json.loads(Path('part3/best_rollback.json').read_text())
    checkpoint = Path('outputs/phase2_reproduction/run/train/flan_t5_base/best')
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    assert sha(checkpoint/'model.safetensors') == frozen['weights_sha256']
    config = json.loads(Path('part2/protocol.json').read_text())
    # FP32 and the original batch size preserve the original inference settings.
    args.out.mkdir(parents=True)
    sources = {
        'id_candidates': 'test_id_candidates_traces.jsonl',
        'ood_candidates': 'test_ood_candidates_traces.jsonl',
        'generated': 'test_id_generated_traces.jsonl',
    }
    base = Path('outputs/phase2_feedback/run_001')
    write_json(args.out/'protocol.json', {
        'checkpoint_sha256': frozen['weights_sha256'], 'frozen_r': frozen,
        'device': 'cuda', 'precision': config['training']['precision'],
        'sources': {k: {'path': str(base/v), 'sha256': sha(base/v)} for k,v in sources.items()},
        'v': 'fresh GPU inference; vanilla and structured prompts; same frozen weights',
        'guards': ['model', 'vocabulary', 'terminal', 'trace'],
        'limits': 'structured prompt is inference-only, not matched retraining; exploratory reused test sets; no test selection',
        'r_limits': 'same model and constrained R in all groups; prefix legality is a safety rejection, never oracle relocation; terminal guard includes vocabulary check',
        'code_sha256': {f:sha(f) for f in [__file__, 'ppnl/verifier_guard.py', 'ppnl/repair_variants.py', 'part2/run_rollback.py']},
    })
    results = {}
    started = time.monotonic()
    with gpu_budget(args.out/'gpu_ledger.json', 2, 'fixed-R verifier-rule paired evaluation') as check:
        runner = Seq2Seq(str(checkpoint), config)
        scorer = NeuralPolicy(runner, check)
        for pool, filename in sources.items():
            rows = [r for r in read_jsonl(base/filename) if r['method']=='self']
            assert len(rows)==200 and len({r['id'] for r in rows})==200
            grids = [Grid.from_dict(r['grid']) for r in rows]
            candidates = [r['candidate'] for r in rows]
            results[pool] = {'direct': plan_metrics(grids, candidates)}
            for style in ['vanilla', 'structured']:
                print('V inference', pool, style, flush=True)
                predictions = runner.predict([prompt('verification', g, c, input_format=style)
                                               for g,c in zip(grids,candidates)], check)
                write_jsonl(args.out/f'{pool}_{style}_predictions.jsonl', [
                    {'id':r['id'], 'prediction':v, 'previous_prediction':r['model_feedback']}
                    for r,v in zip(rows,predictions)])
                for mode in ['model', 'vocabulary', 'terminal', 'trace']:
                    name = f'{style}_{mode}'
                    outputs, feedbacks, traces = [], [], []
                    for r,g,c,v in zip(rows,grids,candidates,predictions):
                        check()
                        fb = guard_verdict(g,c,v,mode)
                        out,status = run_repair(g,c,fb,scorer,frozen)
                        # Oracle labels and BFS metrics are used only after inference.
                        gold, final = execute(g,c), execute(g,out)
                        outputs.append(out)
                        feedbacks.append(fb)
                        traces.append({'id':r['id'], 'grid':r['grid'], 'candidate':c,
                                       'model_feedback':v, 'feedback':fb, 'output':out,
                                       'status':status, 'gold':gold.label(), 'final':final.label()})
                    truth = [execute(g,c) for g,c in zip(grids,candidates)]
                    bad = [i for i,t in enumerate(truth) if not t.valid]
                    good = [i for i,t in enumerate(truth) if t.valid]
                    labels = [{'valid':t.valid,'step':t.step,'reason':t.reason} for t in truth]
                    parsed = [parse_verdict(f) for f in feedbacks]
                    results[pool][name] = {
                        'planning': plan_metrics(grids,outputs),
                        'invalid_only': plan_metrics([grids[i] for i in bad], [outputs[i] for i in bad]),
                        'verification': verification_metrics(labels,feedbacks),
                        'false_accepts': sum(parsed[i] is not None and parsed[i][0] for i in bad),
                        'false_rejects': sum(parsed[i] is not None and not parsed[i][0] for i in good),
                        'valid_routes_broken': sum(not execute(grids[i],outputs[i]).valid for i in good),
                        'statuses':dict(Counter(t['status'] for t in traces)),
                        'v_prediction_matches_previous':sum(v==r['model_feedback'] for r,v in zip(rows,predictions)),
                    }
                    write_jsonl(args.out/f'{pool}_{name}_traces.jsonl',traces)
                    write_json(args.out/'results.json',results)
                    print(pool,name,results[pool][name]['planning'],flush=True)
        write_json(args.out/'completed.json',{'seconds':time.monotonic()-started,'usage':scorer.usage,
                                             'cuda_peak_bytes':runner.torch.cuda.max_memory_allocated(),
                                             'training_updates':0})


if __name__ == '__main__':
    main()
