"""Audit lexical shortcuts and fit equal-budget text-only verifier controls."""

import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import random
import time
import zipfile

import torch
from torch.nn import functional as F
from peft import LoraConfig, PeftModel, get_peft_model

from ppnl.core import ACTIONS, Grid, execute, parse_verdict
from ppnl.data import read_jsonl, write_json, write_jsonl
from ppnl.experiment_freeze import file_hash
from ppnl.modeling import Seq2Seq, gpu_budget
from ppnl.policy import neighbors
from run_prompt_study import collect_queries, aggregate, stats
from state_baselines.step_verification import step_prompt
from ppnl.core import OfficialOracle


CLASSES=("valid","goal_not_reached","invalid_action_token","obstacle_collision","out_of_bounds")


def category(target):
    return "valid" if target=="valid" else parse_verdict(target)[2]


def local_gold(g,p,a):
    if a is None:
        return "valid" if p==g.goal else "invalid | step=1 | reason=goal_not_reached"
    if a not in ACTIONS:
        return "invalid | step=1 | reason=invalid_action_token"
    dr,dc=ACTIONS[a]
    dest=(p[0]+dr,p[1]+dc)
    reason="out_of_bounds" if not g.inside(dest) else "obstacle_collision" if dest in g.obstacles else None
    if reason:
        return f"invalid | step=1 | reason={reason}"
    return "valid" if dest==g.goal else "invalid | step=2 | reason=goal_not_reached"


def render(g,p,a,style="original"):
    base=step_prompt(g,p,a)
    if style=="original":
        return base
    parts=base.split("\n")
    fields=[
        f"Current position: {tuple(p)}.", f"Goal: {g.goal}.",
        f"Obstacles: {list(reversed(g.obstacles))}.",f"Candidate action: {json.dumps(a)}.",
    ]
    if style=="reorder":
        return parts[0]+"\n"+"\n".join(fields[i] for i in (3,2,1,0))+"\nA null action means stop without moving.\n"+parts[3]
    return parts[0]+"\n"+" ".join(fields[i] for i in (1,0,3,2))+"\nA null action means stop without moving.\n"+parts[3]


def pack(g,p,a,identifier):
    from dataclasses import asdict
    y=local_gold(g,p,a)
    return {"id":identifier,"task":"verification","grid":asdict(g),
            "verification_position":list(p),"action":a,"input":step_prompt(g,p,a),"target":y}


def balanced_data(source,seed):
    rng=random.Random(seed)
    grids=list(dict.fromkeys(Grid.from_dict(r["grid"]) for r in source))
    rng.shuffle(grids)
    buckets=defaultdict(list)
    for g in grids:
        states=[(r,c) for r in range(g.rows) for c in range(g.cols) if (r,c) not in g.obstacles]
        rng.shuffle(states)
        for p in states:
            for a in (*ACTIONS,None):
                key=category(local_gold(g,p,a))
                sub="null" if a is None else "move"
                buckets[(key,sub)].append((g,p,a))
        for token in ("jump","north","wait","diagonal","stay","UP","forward","turn"):
            buckets[("invalid_action_token","move")].append((g,rng.choice(states),token))
    selected=[]
    for kind in CLASSES:
        groups=(("null",500),("move",500)) if kind=="valid" else (("null",125),("move",125)) if kind=="goal_not_reached" else (("move",250),)
        for subgroup,count in groups:
            pool=buckets[(kind,subgroup)]
            rng.shuffle(pool)
            # Different states/actions, repeated layouts allowed solely within train.
            selected.extend(pool[:count])
    assert len(selected)==2000
    rng.shuffle(selected)
    return [pack(g,p,a,f"balanced-{i}") for i,(g,p,a) in enumerate(selected)]


def probes(dev_rows):
    grids=list(dict.fromkeys(Grid.from_dict(r["grid"]) for r in dev_rows))
    rows=[]
    counts=Counter()
    for g in grids:
        for p in ((r,c) for r in range(g.rows) for c in range(g.cols) if (r,c) not in g.obstacles):
            for action,dest in neighbors(g,p).items():
                if dest==g.goal and p!=g.start and counts["goal"]<50:
                    alt=next(q for q in neighbors(g,p).values() if q not in (g.start,g.goal)) if any(
                        q not in (g.start,g.goal) for q in neighbors(g,p).values()) else None
                    if alt:
                        changed=Grid(g.rows,g.cols,g.obstacles,g.start,alt)
                        rows.extend((dict(type="goal",pair=counts["goal"],**pack(g,p,action,f"goal-{counts['goal']}-a")),
                                     dict(type="goal",pair=counts["goal"],**pack(changed,p,action,f"goal-{counts['goal']}-b"))))
                        counts["goal"]+=1
                if dest not in (g.start,g.goal) and p!=g.goal and counts["obstacle"]<50:
                    changed=Grid(g.rows,g.cols,(*g.obstacles,dest),g.start,g.goal)
                    rows.extend((dict(type="obstacle",pair=counts["obstacle"],**pack(g,p,action,f"obstacle-{counts['obstacle']}-a")),
                                 dict(type="obstacle",pair=counts["obstacle"],**pack(changed,p,action,f"obstacle-{counts['obstacle']}-b"))))
                    counts["obstacle"]+=1
            if p[0]==0 and (1,p[1]) not in g.obstacles and counts["boundary"]<50:
                other=(1,p[1])
                rows.extend((dict(type="boundary",pair=counts["boundary"],**pack(g,p,"up",f"boundary-{counts['boundary']}-a")),
                             dict(type="boundary",pair=counts["boundary"],**pack(g,other,"up",f"boundary-{counts['boundary']}-b"))))
                counts["boundary"]+=1
            if p!=g.goal and counts["stop"]<50:
                rows.extend((dict(type="stop",pair=counts["stop"],**pack(g,g.goal,None,f"stop-{counts['stop']}-a")),
                             dict(type="stop",pair=counts["stop"],**pack(g,p,None,f"stop-{counts['stop']}-b"))))
                counts["stop"]+=1
        if all(v==50 for v in counts.values()) and len(counts)==4:
            break
    assert len(rows)==400 and all(v==50 for v in counts.values())
    assert all(rows[i]["target"]!=rows[i+1]["target"] for i in range(0,400,2))
    return rows


def lexical_stats(rows):
    counts=Counter(category(r["target"]) for r in rows)
    action=defaultdict(Counter)
    length=defaultdict(Counter)
    for r in rows:
        a="null" if r["action"] is None else r["action"]
        y=category(r["target"])
        action[a][y]+=1
        length[len(r["input"])//40][y]+=1
    return {"n":len(rows),"counts":dict(counts),"fractions":{k:v/len(rows) for k,v in counts.items()},
            "action_label":{k:dict(v) for k,v in action.items()},
            "length_bucket_label":{k:dict(v) for k,v in length.items()},
            "unique_inputs":len(set(r["input"] for r in rows)),
            "input_contains_id":sum(r["id"] in r["input"] for r in rows)}


def loss_for(runner,batch,weights):
    labels=batch["labels"]
    output=runner.model(**{k:v for k,v in batch.items() if k!="policy_actions"})
    ce=F.cross_entropy(output.logits.float().transpose(1,2),labels,ignore_index=-100,reduction="none")
    lengths=(labels!=-100).sum(-1)
    per=.5*ce[:,0]+.5*ce[:,1:].sum(-1)/(lengths-1).clamp_min(1)
    return (per*weights).mean()


def main():
    p=argparse.ArgumentParser()
    for k in ("checkpoint","control","previous","evidence","out","executor","ledger"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path("state_baselines/protocol.json").read_text())
    config["training"]["eval_batch_size"]=64
    oracle=OfficialOracle(a.executor)
    with zipfile.ZipFile(a.evidence) as archive:
        original=[json.loads(s) for s in archive.read("v1/train.jsonl").splitlines()]
        v2=[json.loads(s) for s in archive.read("v2/train.jsonl").splitlines()]
    balanced=balanced_data(original,42)
    groups={g:read_jsonl(a.previous/f"candidates_{g}.jsonl") for g in (
        "dev","dev_generated","test_id","test_ood","g_peer_old","g_peer_new","g_old","g_new")}
    probeset=probes(groups["dev"])
    hold={Grid.from_dict(r["grid"]).layout_id for rs in (*groups.values(),probeset) for r in rs}
    assert not hold.intersection(Grid.from_dict(r["grid"]).layout_id for r in balanced)
    raw_v2=[r for r in v2 if r["task"]=="verification"]
    report={"source":file_hash(__file__),"checkpoint":file_hash(a.checkpoint/"model.safetensors"),
            "data":{"original":lexical_stats(original),"v2_direct":lexical_stats(raw_v2),"balanced":lexical_stats(balanced)},
            "arms":{}}
    for name,rs in (("original",original),("balanced",balanced),("probes",probeset)):
        write_jsonl(a.out/f"{name}.jsonl",rs)
    for g,rs in groups.items():
        write_jsonl(a.out/f"candidates_{g}.jsonl",rs)
    pools={g:collect_queries(rs) for g,rs in groups.items()}
    queries=list(dict.fromkeys(q for qs,_ in pools.values() for q in qs))
    texts={style:[render(*q,style) for q in queries] for style in ("original","reorder","reflow")}
    write_json(a.out/"data_audit.json",report["data"])
    def evaluate(runner,name,history=None):
        arm={"metrics":{},"robustness":{},"probes":{},"history":history}
        for style in texts:
            preds=runner.predict(texts[style],check)
            cache=dict(zip(queries,preds,strict=True))
            for g,(qs,paths) in pools.items():
                pred=aggregate([cache[q] for q in qs],paths,groups[g])
                m,rs=stats(groups[g],pred,oracle,True)
                key="metrics" if style=="original" else "robustness"
                arm[key].setdefault(style,{})[g]=m
                write_jsonl(a.out/f"{name}_{style}_{g}.jsonl",rs)
            ps=runner.predict([render(Grid.from_dict(r["grid"]),r["verification_position"],r["action"],style)
                               for r in probeset],check)
            detail={}
            for kind in ("goal","obstacle","boundary","stop"):
                indices=[i for i,r in enumerate(probeset) if r["type"]==kind]
                detail[kind]={"n":100,
                    "exact":sum(ps[i].strip()==probeset[i]["target"] for i in indices),
                    "both_exact_pairs":sum(ps[i].strip()==probeset[i]["target"] and ps[i+1].strip()==probeset[i+1]["target"]
                                           for i in indices[::2]),
                    "unchanged_pairs":sum(ps[i].strip()==ps[i+1].strip() for i in indices[::2])}
            arm["probes"][style]=detail
            write_jsonl(a.out/f"{name}_{style}_probes.jsonl",[
                {"id":r["id"],"type":r["type"],"target":r["target"],"prediction":pred}
                for r,pred in zip(probeset,ps,strict=True)])
            print(name,style,"done",flush=True)
        report["arms"][name]=arm
        write_json(a.out/"results.json",report)
    with gpu_budget(a.ledger,20,"Hierarchical class balance and shortcut study") as check:
        control=Seq2Seq(str(a.checkpoint),config)
        control.model=PeftModel.from_pretrained(control.model,a.control)
        evaluate(control,"control_original")
        del control
        gc.collect()
        configurations=(("hierarchical_weighted",original,False,42,True),
                        ("hierarchical_data",balanced,False,42,False),
                        ("hierarchical_formats",balanced,True,42,False),
                        ("hierarchical_formats_seed43",balanced,True,43,False))
        for name,source,mix,seed,reweight in configurations:
            rows=[]
            for i,r in enumerate(source):
                style=("original","reorder","reflow")[i%3] if mix else "original"
                rows.append({**r,"input":render(Grid.from_dict(r["grid"]),r["verification_position"],r["action"],style)})
            runner=Seq2Seq(str(a.checkpoint),config)
            torch.manual_seed(seed)
            runner.model=get_peft_model(runner.model,LoraConfig(task_type="SEQ_2_SEQ_LM",r=8,lora_alpha=16,
                lora_dropout=0.0,target_modules=["q","v"]))
            features=runner.encode_rows(rows)
            counts=Counter(category(r["target"]) for r in rows)
            weights=torch.tensor([(1 if category(r["target"])=="valid" else len(rows)/(8*counts[category(r["target"])])) if reweight else 1 for r in rows],
                                 device=runner.device)
            parameters=[p for p in runner.model.parameters() if p.requires_grad]
            optimizer=torch.optim.AdamW(parameters,lr=3e-4)
            best=None
            logs=[]
            loader=torch.utils.data.DataLoader(
                list(range(len(rows))),batch_size=4,shuffle=True,collate_fn=list,
                generator=torch.Generator().manual_seed(seed))
            for epoch in (1,2):
                runner.model.train()
                optimizer.zero_grad(set_to_none=True)
                total=0
                for n,ids in enumerate(loader,1):
                    check()
                    batch=runner.collate([features[j] for j in ids]).to(runner.device)
                    loss=loss_for(runner,batch,weights[ids])
                    assert torch.isfinite(loss)
                    total+=float(loss.detach())
                    (loss/4).backward()
                    if n%4==0:
                        torch.nn.utils.clip_grad_norm_(parameters,1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                dev={}
                # Includes formatting holdouts from dev only, never test selection.
                for style in ("original","reorder"):
                    for g in ("dev","dev_generated"):
                        qs,paths=pools[g]
                        outputs=runner.predict([render(*q,style) for q in qs],check)
                        m,_=stats(groups[g],aggregate(outputs,paths,groups[g]),oracle,False)
                        dev[f"{style}_{g}"]=m
                rank=(sum(m["macro_f1"] for m in dev.values())/4,
                      -sum(m["false_reject"] for m in dev.values()),
                      dev["original_dev"]["joint_error_accuracy"])
                logs.append({"epoch":epoch,"rank":rank,"loss":total/500,"dev":dev})
                print(name,epoch,rank,flush=True)
                if best is None or rank>best[0]:
                    best=(rank,epoch,{n:p.detach().cpu().clone() for n,p in runner.model.named_parameters()
                                     if p.requires_grad})
            runner.model.load_state_dict(best[2],strict=False)
            runner.model.save_pretrained(a.out/f"adapter_{name}")
            evaluate(runner,name,{"epochs":logs,"selected":best[1],"updates":250,"seed":seed})
            del runner,optimizer
            gc.collect()
    write_json(a.out/"completed.json",{"arms":list(report["arms"])})


if __name__=="__main__":
    main()
