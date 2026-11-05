"""Exploratory CoT SFT + outcome GRPO; synthetic BFS feedback, not RLHF."""
import json
import random
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ppnl.core import Grid, ACTIONS, shortest_path
from ppnl.data import read_jsonl, write_json
from ppnl.modeling import Seq2Seq, gpu_budget, require_cuda
from ppnl.task_adapters import lora_config, checkpoint_hash
from ppnl.rlvr import completion_mask, group_advantages, grpo_objective
from reachability_probe import load_cases, score, reward

def query(g):
    return g.description() + ' Track coordinates. End with FINAL: followed by moves or NO_PATH.'

def answer(text):
    # A missing or repeated delimiter is invalid, not silently repaired.
    return text.split('FINAL:')[1].strip() if text.count('FINAL:') == 1 else ''

def target(g):
    path = shortest_path(g)
    if path is None:
        seen = {g.start}
        todo = [g.start]
        for p in todo:
            for dr, dc in ACTIONS.values():
                q = (p[0]+dr, p[1]+dc)
                if g.inside(q) and q not in g.obstacles and q not in seen:
                    seen.add(q)
                    todo.append(q)
        trace = 'reachable ' + ' '.join(f'{r},{c}' for r,c in sorted(seen))
        return trace + ' FINAL: NO_PATH'
    p = g.start
    trace = [f'{p[0]},{p[1]}']
    for a in path:
        dr,dc = ACTIONS[a]
        p = p[0]+dr,p[1]+dc
        trace.append(f'{p[0]},{p[1]}')
    return 'trace ' + ' '.join(trace) + ' FINAL: ' + ' '.join(path)

def main():
    import torch
    from peft import get_peft_model
    out = ROOT/'outputs/reachability_cot_grpo_001'
    if out.exists(): raise FileExistsError(out)
    torch = require_cuda()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    random.seed(42)
    batches = {s:read_jsonl(ROOT/f'data/reachability_feedback_pending/{s}.jsonl') for s in ('train','dev')}
    grids = {s:{r['id']:Grid.from_dict(r['grid']) for r in rows} for s,rows in batches.items()}
    probes = load_cases()[0]
    used = set()
    for group in (*grids.values(),probes):
        ids = {g.layout_id for g in group.values()}
        assert not used & ids
        used.update(ids)
    old = {r['layout_id'] for s in ('train','dev','test_id','test_ood') for r in read_jsonl(ROOT/f'data/derived/{s}.jsonl')}
    assert not old & {g.layout_id for group in grids.values() for g in group.values()}
    config=json.loads((ROOT/'artifacts/campaign_20261103/baselines.json').read_text())
    config['local_files_only']=True
    config['training']['max_target_tokens']=256
    base=ROOT/'artifacts/campaign_20261103/baselines/t5_base/train/best'
    out.mkdir()
    write_json(out/'protocol.json',{'seed':42,'train':16,'dev':8,'sft_passes':3,'rl_groups':16,
        'group_size':4,'updates_per_rollout':2,'human_feedback':False,'base_sha256':checkpoint_hash(base),
        'scope':'exploratory extra-data pilot; not a compliant replacement for the 4000-example main run',
        'selection':'retain SFT unless GRPO strictly improves balanced dev outcome score',
        'reasoning_evaluation':'score final outcome only; generated rationale is not certified'})
    with gpu_budget(config['gpu_ledger'],config['gpu_hours_cap'],'reachability:cot_grpo') as check:
        runner=Seq2Seq(str(base),config)
        runner.model=get_peft_model(runner.model,lora_config({'r':8,'alpha':16,'dropout':0.0}))
        model=runner.model
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4)
        def evaluate(group):
            text=runner.predict([query(g) for g in group.values()],check)
            predictions=dict(zip(group,map(answer,text)))
            metrics=score(group,predictions)
            metrics['format_valid']=sum(t.count('FINAL:')==1 for t in text)
            return {'metrics':metrics,'responses':dict(zip(group,text))}
        reports={'baseline_dev':evaluate(grids['dev'])}
        rows=[{'id':k,'input':query(g),'target':target(g)} for k,g in grids['train'].items()]
        runner.check_lengths(rows)
        for epoch in range(3):
            random.shuffle(rows)
            for offset in range(0,len(rows),4):
                check()
                model.train()
                batch=runner.collate(rows[offset:offset+4]).to('cuda')
                loss=model(**batch).loss
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
            print('SFT epoch',epoch+1,flush=True)
        model.save_pretrained(out/'sft')
        runner.tokenizer.save_pretrained(out/'sft')
        reports['sft_dev']=evaluate(grids['dev'])
        reports['sft_probe']=evaluate(probes)
        for group in optimizer.param_groups: group['lr']=5e-5
        rollouts=[]
        for step,g in enumerate(grids['train'].values(),1):
            check(); model.eval()
            encoded=runner.tokenizer([query(g)]*4,return_tensors='pt').to('cuda')
            with torch.no_grad():
                sampled=model.generate(**encoded,do_sample=True,top_k=0,top_p=1.,max_new_tokens=256,num_beams=1)
            texts=runner.tokenizer.batch_decode(sampled,skip_special_tokens=True)
            rewards=[reward(g,answer(t)) for t in texts]
            labels=sampled[:,1:]
            mask=completion_mask(labels,runner.tokenizer.eos_token_id,runner.tokenizer.pad_token_id)
            batch=dict(encoded,decoder_input_ids=sampled[:,:-1])
            with torch.no_grad(),model.disable_adapter():
                ref=model(**batch).logits.log_softmax(-1).gather(-1,labels[...,None]).squeeze(-1)
            with torch.no_grad():
                oldlp=model(**batch).logits.log_softmax(-1).gather(-1,labels[...,None]).squeeze(-1)
            adv=group_advantages(rewards).to('cuda')
            for _ in range(2):
                check()
                lp=model(**batch).logits.log_softmax(-1).gather(-1,labels[...,None]).squeeze(-1)
                loss=grpo_objective(lp,oldlp,ref,mask,adv,.02)
                if not torch.isfinite(loss): raise FloatingPointError('RL loss')
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
            rollouts.append({'step':step,'responses':texts,'rewards':rewards,'advantages':adv.tolist()})
            print('GRPO group',step, rewards,flush=True)
        reports['grpo_dev']=evaluate(grids['dev'])
        # No adaptation follows diagnostic evaluation.
        reports['grpo_probe']=evaluate(probes)
        quality=lambda r:r['metrics']['unreachable_detected']+r['metrics']['reachable_solved']
        reports['selected']='grpo' if quality(reports['grpo_dev'])>quality(reports['sft_dev']) else 'sft'
        model.save_pretrained(out/'grpo')
        runner.tokenizer.save_pretrained(out/'grpo')
        write_json(out/'rollouts.json',rollouts)
        write_json(out/'report.json',reports)
        print(json.dumps({k:v['metrics'] for k,v in reports.items() if isinstance(v,dict)}),flush=True)

if __name__=='__main__': main()
