"""Controlled from-pretrained shared G/C/V experiment with four-action rewards."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from hope_joint_data import (CODE_COMMIT, DATA_COMMIT, LABELS, MOVES, category, fields,
                             prepare, read, rows, save, sha, write_rows)


def metrics(gold, pred, labels):
    counts = Counter(zip(gold, pred))
    per = {}
    for label in labels:
        n = sum(a == label for a in gold)
        p = sum(a == label for a in pred)
        tp = counts[label, label]
        per[label] = dict(n=n, correct=tp, predicted=p, false_positive=p-tp,
                          recall=tp/n if n else None, f1=2*tp/(n+p) if n+p else 0)
    return dict(n=len(gold), correct=sum(a == b for a, b in zip(gold, pred)),
                accuracy=sum(a == b for a, b in zip(gold, pred))/len(gold),
                macro_f1=sum(v['f1'] for v in per.values())/len(labels), classes=per)


def step_evaluate(net, data, folder):
    folder = Path(folder)
    was_training = net.model.training
    net.model.eval()
    records, report = [], {}
    for task in ('G', 'C', 'V'):
        group = [r for r in data if r['task'] == task]
        outputs = net.predict(task, [r['text'] for r in group])
        correct, errors, direction = 0, Counter(), {}
        task_records = []
        for row, value in zip(group, outputs):
            prediction = value[0] if task == 'G' else value
            ok = prediction in row['answers']
            correct += ok
            item = dict(id=row['id'], task=task, gold=row['answers'], prediction=prediction, correct=ok)
            if task == 'G':
                current, goal, walls = fields(row['text'])
                label, _ = category(current, goal, walls, prediction)
                item['move_result'] = label
                if not ok:
                    errors[label if label in ('out_of_bounds', 'obstacle_collision') else 'legal_detour'] += 1
                if len(row['answers']) == 1:
                    cell = direction.setdefault(row['answers'][0], dict(n=0, correct=0))
                    cell['n'] += 1
                    cell['correct'] += ok
            task_records.append(item)
        report[task] = dict(n=len(group), correct=correct, accuracy=correct/len(group))
        if task == 'G':
            report[task].update(errors=dict(errors), unique_directions=direction)
        if task == 'V':
            report[task].update(metrics([r['answers'][0] for r in group], outputs, LABELS))
        records.extend(task_records)
    write_rows(folder / 'predictions.jsonl', records)
    save(folder / 'metrics.json', report)
    if was_training:
        net.model.train()
    return report


def train_arm(args, arm, corpus, rewards):
    import torch
    import torch.nn.functional as F
    from experiments.part1_baseline.model import Network
    from experiments.part1_baseline.protocol import MODELS, schedule
    arm_dir = args.out / arm
    if (arm_dir / 'train_done.json').exists():
        done = read(arm_dir / 'train_done.json')
        assert sha(arm_dir / 'model/model.safetensors') == done['weights']
        return done
    if (arm_dir / 'started.json').exists():
        raise RuntimeError('Incomplete run already exists; use a fresh output directory: ' + str(arm_dir))
    assert sha(args.base / 'model.safetensors') == MODELS['flan_t5_base'][2]
    assert sha(args.base / 'tokenizer.json') == MODELS['flan_t5_base'][3]
    assert (args.base / 'upstream_revision.txt').read_text().strip() == MODELS['flan_t5_base'][1]
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    net = Network(args.base)
    model = net.model
    assert all(p.requires_grad for p in model.parameters())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    encoded = net.encode([r['text'] for r in corpus])
    choices, classes = [], []
    for row in corpus:
        cls = LABELS.index(row['answers'][0]) if row['task'] == 'V' else None
        classes.append(cls)
        choices.append([(net.label_ids[cls],)] if cls is not None else [net.target(a) for a in row['answers']])
    batches, weights, normalizer = schedule(corpus)
    assert len(batches) == 375
    w = torch.tensor(weights, device='cuda', dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01, fused=True)
    save(arm_dir / 'started.json', dict(arm=arm, code_commit=CODE_COMMIT, dataset_commit=DATA_COMMIT,
         seed=42, pretrained=sha(args.base/'model.safetensors'), updates=375, shared_all_parameters_trainable=True,
         parameters=sum(p.numel() for p in model.parameters()), lr=1e-4, microbatch=args.microbatch,
         script_sha256=sha(__file__), reward_strength=.2 if arm!='supervised' else 0,
         four_answer_extra_forward='performed in every arm inside fork_rng; zero gradient coefficient in supervised arm',
         no_cv_freezing=True, four_action_order=MOVES,
         batch_order_sha256=hashlib.sha256(json.dumps([b for _,_,b in batches],separators=(',',':')).encode()).hexdigest()))
    dev = rows(args.repo / 'data/coordinate_gcv_review/dev.jsonl')
    model.train()
    start = time.perf_counter()
    reward_mass, unsafe_mass, samples, varied = 0., 0., 0, 0
    for k, (stream, idx, batch) in enumerate(batches):
        torch.manual_seed(42 + idx + (10000 if stream == 'GV' else 0))
        opt.zero_grad(set_to_none=True)
        supervised_total = reward_total = 0.
        for offset in range(0, len(batch), args.microbatch):
            group = batch[offset:offset+args.microbatch]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                terms = net.loss([encoded[j] for j in group], [choices[j] for j in group], [classes[j] for j in group])
                main = terms.dot(w[group]) / normalizer
            main.backward()
            supervised_total += main.detach().item()
            generators = [j for j in group if corpus[j]['task'] == 'G']
            if generators:
                # Preserve the exact RNG stream used by the original supervised recipe.
                with torch.random.fork_rng(devices=[0]):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits, target = net.logits([encoded[j] for j in generators], [net.moves]*len(generators))
                        scores = -F.cross_entropy(logits.float().transpose(1,2), target, reduction='none').sum(1)
                        scores = scores.reshape(-1, 4)
                        logp = scores.log_softmax(1)
                        probability = logp.exp()
                        rv = scores.new_tensor([rewards[corpus[j]['id']] for j in generators])
                        if arm == 'rloo':
                            picked = torch.multinomial(probability.detach(), 8, replacement=True)
                            observed = rv.gather(1, picked)
                            advantage = observed - (observed.sum(1,keepdim=True)-observed)/7
                            extra_terms = -(advantage.detach()*logp.gather(1,picked)).mean(1)
                            varied += int((observed.max(1).values != observed.min(1).values).sum())
                        else:
                            extra_terms = -(probability * rv).sum(1)
                        extra = (.2 if arm != 'supervised' else 0.) * extra_terms.dot(w[generators]) / normalizer
                    extra.backward()
                    reward_total += extra.detach().item()
                    reward_mass += float((probability.detach()*rv).sum())
                    unsafe_mass += float((probability.detach()*(rv == -5)).sum())
                    samples += len(generators)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        if k < 3 or (k+1)%10 == 0 or (k+1)%125 == 0:
            progress = dict(arm=arm, update=k+1, stream=stream, supervised_loss=supervised_total,
                            reward_loss=reward_total, grad_norm=float(norm), elapsed_s=time.perf_counter()-start,
                            mean_reward=reward_mass/max(samples,1), unsafe_probability=unsafe_mass/max(samples,1),
                            varied_sampling_groups=varied, G_presentations=samples,
                            peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
            save(arm_dir/'progress.json', progress)
            print(json.dumps(progress), flush=True)
        if (k+1)%125 == 0:
            result = step_evaluate(net, dev, arm_dir/f'update{k+1}/step_dev')
            print(json.dumps(dict(arm=arm, update=k+1, step_dev=result)), flush=True)
    opt.zero_grad(set_to_none=True)
    net.save(args.base, arm_dir/'model')
    done = dict(arm=arm, updates=375, weights=sha(arm_dir/'model/model.safetensors'),
                elapsed_s=time.perf_counter()-start, G_presentations=samples,
                varied_sampling_groups=varied, all_parameters_trained=True)
    save(arm_dir/'train_done.json', done)
    del net, model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return done


def evaluate_arm(args, arm, sources):
    import torch
    from experiments.part1_baseline.model import Network
    from experiments.part1_baseline import protocol
    from experiments.part1_baseline.evaluate import evaluate
    done = read(args.out / arm / 'train_done.json')
    folder = args.out / arm
    work = folder/'evaluation'
    protocol.SOURCES.update(sources)
    if not (work/'data/protocol.json').exists():
        shutil.copytree(args.out/'data', work/'data')
    net = Network(folder/'model')
    net.model.eval()
    weight = done['weights']
    dev = evaluate(net, work, 'dev', arm, weight, False)
    # Each arm has a fixed final checkpoint; all arms are declared before test access.
    decision = dict(protocol=sha(work/'data/protocol.json'), primary_model=arm,
                    scores={arm:dict(weights=weight)}, fixed_checkpoint=True,
                    note='No checkpoint or reward selection uses double_error or single_error outcomes.')
    save(work/'selection.json', decision)
    scores = dict(dev=dev)
    for split in ('double_error', 'single_error'):
        scores[split] = evaluate(net, work, split, arm, weight, True)
    save(folder/'all_metrics.json', scores)
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return scores


def compare(args):
    all_metrics = {arm:read(args.out/arm/'all_metrics.json') for arm in args.arms}
    compact = {}
    for arm, groups in all_metrics.items():
        compact[arm] = {'step_dev':read(args.out/arm/'update375/step_dev/metrics.json')}
        for split, result in groups.items():
            compact[arm][split] = dict(verification_exact=result['verification']['exact'],
                verification_n=result['verification']['n'], macro_f1=result['verification']['structured_macro_f1'],
                repair_success=result['candidate_repair']['success'], direct_success=result['pipeline']['direct']['success'],
                self_repair_success=result['pipeline'].get('self_repair',{}).get('success'))
    save(args.out/'comparison.json', compact)
    print(json.dumps(dict(comparison=compact)), flush=True)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--repo', type=Path, required=True)
    cli.add_argument('--dataset', type=Path, required=True)
    cli.add_argument('--base', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True)
    cli.add_argument('--arms', nargs='+', default=['supervised','expected','rloo'], choices=['supervised','expected','rloo'])
    cli.add_argument('--data-only', action='store_true')
    cli.add_argument('--train-only', action='store_true')
    cli.add_argument('--microbatch', type=int, default=16)
    args = cli.parse_args()
    sys.path.insert(0,str(args.repo))
    if args.train_only:
        corpus = rows(args.repo/'data/coordinate_gcv_review/train.jsonl')
        from hope_joint_data import reward_tables
        rewards, _, _ = reward_tables(corpus)
        sources = None
    else:
        corpus, rewards, _, sources = prepare(args.repo,args.dataset,args.out)
    if args.data_only:
        print(json.dumps(read(args.out/'data_audit.json')),flush=True)
        return
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    if args.train_only:
        for arm in args.arms:
            train_arm(args,arm,corpus,rewards)
        return
    # Parallel workers fit comfortably in H100 memory and share no mutable model state.
    workers = []
    for arm in args.arms:
        if (args.out/arm/'train_done.json').exists():
            continue
        folder = args.out/arm
        folder.mkdir(exist_ok=True,parents=True)
        command = [sys.executable,'-u',__file__,'--repo',str(args.repo),'--dataset',str(args.dataset),
                   '--base',str(args.base),'--out',str(args.out),'--arms',arm,'--train-only',
                   '--microbatch',str(args.microbatch)]
        process = subprocess.Popen(command,stdout=(folder/'train.log').open('ab',buffering=0),
                                   stderr=subprocess.STDOUT)
        workers.append((arm,process))
    for arm, process in workers:
        if process.wait() != 0:
            raise RuntimeError('Training worker failed: '+arm)
    save(args.out/'training_complete.json',dict(arms=args.arms,code_commit=CODE_COMMIT,dataset_commit=DATA_COMMIT))
    for arm in args.arms:
        evaluate_arm(args,arm,sources)
    compare(args)
    save(args.out/'completed.json',dict(code_commit=CODE_COMMIT,dataset_commit=DATA_COMMIT,
                                      arms=args.arms,seed=42,claim='single-seed controlled pilot'))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
