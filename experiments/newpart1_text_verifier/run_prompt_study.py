"""Only change V prompt text; hold data, loss, model, and scanner fixed."""

import argparse
import gc
import json
from pathlib import Path
import time
import zipfile

import torch
from peft import LoraConfig, get_peft_model

from ppnl.core import Grid, OfficialOracle, execute, parse_verdict
from ppnl.data import read_jsonl, write_json, write_jsonl
from ppnl.evaluation import plan_metrics, verification_metrics
from ppnl.experiment_freeze import file_hash
from ppnl.modeling import Seq2Seq, gpu_budget
from ppnl.policy import neighbors
from state_baselines.step_verification import step_prompt, verify_steps
from state_baselines.workflow import repair


TEMPLATES = ("github", "compact", "explicit")


def template(grid, position, action, style):
    if style == "github":
        return step_prompt(grid, position, action)
    fields = (
        f"Grid: {grid.rows}x{grid.cols}\n"
        f"Position: {json.dumps(list(position), separators=(',', ':'))}\n"
        f"Goal: {json.dumps(list(grid.goal), separators=(',', ':'))}\n"
        f"Obstacles: {json.dumps([list(p) for p in sorted(grid.obstacles)], separators=(',', ':'))}\n"
        f"Action: {json.dumps(action)}\n"
    )
    if style == "compact":
        return (
            "Verify one action. Zero-based [row,column]; up decreases row, right increases column.\n"
            + fields +
            "null means stop. Reply valid only for legal arrival or stopping at goal.\n"
            "Otherwise reply invalid | step=N | reason=R.\n"
            "R: invalid_action_token (unknown), out_of_bounds (outside), obstacle_collision (blocked), "
            "goal_not_reached (legal but unfinished).\n"
            "N=1 for illegal action or null; N=2 for legal unfinished move."
        )
    return (
        "Verify exactly one action using zero-based [row,column].\n"
        "up=[-1,0]; down=[1,0]; left=[0,-1]; right=[0,1]. null means stop.\n"
        + fields +
        "Check in order: unknown action; next row/column outside grid; obstacle; arrival.\n"
        "Rows range from 0 to rows-1. Columns range from 0 to columns-1.\n"
        "Return exactly one line:\n"
        "valid\n"
        "invalid | step=1 | reason=invalid_action_token\n"
        "invalid | step=1 | reason=out_of_bounds\n"
        "invalid | step=1 | reason=obstacle_collision\n"
        "If legal but not at goal: invalid | step=2 | reason=goal_not_reached\n"
        "If null and not at goal: invalid | step=1 | reason=goal_not_reached\n"
        "valid applies only to legal arrival or null at goal."
    )


def collect_queries(rows):
    lookup, paths = {}, []
    for row in rows:
        grid = Grid.from_dict(row["grid"])
        position = grid.start
        ids = []
        for move in row["candidate"].split() or [None]:
            ids.append(lookup.setdefault((grid, position, move), len(lookup)))
            position = neighbors(grid, position).get(move, position)
        paths.append(ids)
    return list(lookup), paths


def aggregate(local, paths, rows):
    output = []
    for seq, row in zip(paths, rows, strict=True):
        for offset, index in enumerate(seq):
            parsed = parse_verdict(local[index])
            if parsed is None:
                output.append(local[index])
                break
            valid, _, reason = parsed
            if not valid and reason != "goal_not_reached":
                output.append(f"invalid | step={offset+1} | reason={reason}")
                break
            if offset == len(seq)-1:
                eos = 1 if not row["candidate"].split() else len(seq)+1
                output.append("valid" if valid else f"invalid | step={eos} | reason={reason}")
                break
    return output


def stats(rows, predictions, oracle, full):
    grids = [Grid.from_dict(r["grid"]) for r in rows]
    truths = [oracle.check(g, r["candidate"]) for g, r in zip(grids, rows, strict=True)]
    parsed = [parse_verdict(p) for p in predictions]
    m = verification_metrics([vars(t) for t in truths], predictions)
    m.update(
        invalid=sum(not t.valid for t in truths), valid=sum(t.valid for t in truths),
        false_accept=sum(not t.valid and p is not None and p[0] for t, p in zip(truths, parsed, strict=True)),
        false_reject=sum(t.valid and (p is None or not p[0]) for t, p in zip(truths, parsed, strict=True)),
        by_reason={reason: {
            "n": sum(t.reason == reason for t in truths),
            "joint": sum(t.reason == reason and p == (False,t.step,t.reason)
                         for t,p in zip(truths,parsed,strict=True))}
            for reason in ("goal_not_reached","invalid_action_token","obstacle_collision","out_of_bounds")})
    records = []
    if full:
        for r, g, t, p in zip(rows, grids, truths, predictions, strict=True):
            plan, status = repair(g, r["candidate"], p)
            end = oracle.check(g, plan)
            assert execute(g, plan).label() == end.label()
            records.append({"id":r["id"],"candidate":r["candidate"],"gold":t.label(),
                            "prediction":p,"plan":plan,"status":status,"final":end.label()})
        m.update(fixed=sum(r["gold"]!="valid" and r["final"]=="valid" for r in records),
                 broken=sum(r["gold"]=="valid" and r["final"]!="valid" for r in records),
                 planning=plan_metrics(grids,[r["plan"] for r in records]))
    return m, records


def parity_check(rows, query_pool, oracle):
    local, paths = query_pool
    from ppnl.core import ACTIONS
    answers = []
    for grid, pos, move in local:
        if move is None:
            pred = "valid" if pos == grid.goal else "invalid | step=1 | reason=goal_not_reached"
        elif move not in ACTIONS:
            pred = "invalid | step=1 | reason=invalid_action_token"
        else:
            dr, dc = ACTIONS[move]
            nxt = (pos[0]+dr,pos[1]+dc)
            reason = "out_of_bounds" if not grid.inside(nxt) else "obstacle_collision" if nxt in grid.obstacles else None
            pred = (f"invalid | step=1 | reason={reason}" if reason else
                    "valid" if nxt == grid.goal else "invalid | step=2 | reason=goal_not_reached")
        answers.append(pred)
    result = aggregate(answers, paths, rows)
    assert result == [oracle.check(Grid.from_dict(r["grid"]),r["candidate"]).label() for r in rows]
    mapping = {step_prompt(g,p,a):pred for (g,p,a),pred in zip(local,answers,strict=True)}
    class Fake:
        def predict(self, prompts, check):
            return [mapping[s] for s in prompts]
    assert result == verify_steps(Fake(),[Grid.from_dict(r["grid"]) for r in rows],
                                 [r["candidate"] for r in rows])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint","out","pool","evidence","executor","ledger"):
        p.add_argument("--"+key, type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path("state_baselines/protocol.json").read_text())
    config["training"]["eval_batch_size"] = 64
    oracle = OfficialOracle(a.executor)
    with zipfile.ZipFile(a.evidence) as archive:
        train = [json.loads(s) for s in archive.read("v1/train.jsonl").splitlines()]
    assert len(train) == 2000
    groups = {g:read_jsonl(a.pool/f"candidates_{g}.jsonl") for g in (
        "dev","dev_generated","test_id","test_ood","g_peer_old","g_peer_new","g_old","g_new",
        "7x7_single","7x7_double","dense_6x6_single","dense_6x6_double")}
    heldout = {Grid.from_dict(r["grid"]).layout_id for rs in groups.values() for r in rs}
    assert not heldout.intersection(Grid.from_dict(r["grid"]).layout_id for r in train)
    pools = {g:collect_queries(rs) for g,rs in groups.items()}
    for g in ("dev","test_id","test_ood"):
        parity_check(groups[g],pools[g],oracle)
    summary = {
        "reference_commit":"9eeb494fe1f775ad033ce2f668e22517cd194aee",
        "checkpoint":file_hash(a.checkpoint/"model.safetensors"), "source":file_hash(__file__),
        "training_rows":2000,"epochs":2,"data":"GitHub v1 exact V rows; targets unchanged",
        "loss":"0.5 first-token CE plus 0.5 mean remaining-token CE",
        "inference_batch_size":64,
        "notes":"No geometry, no added STOP; same official scanner and local R",
        "arms":{},"templates":{}}
    for g,rs in groups.items():
        write_jsonl(a.out/f"candidates_{g}.jsonl",rs)
    with gpu_budget(a.ledger,20,"Prompt-only controlled v1 adaptation") as check:
        frozen = Seq2Seq(str(a.checkpoint),config)
        prompt_texts = {}
        for style in TEMPLATES:
            prompt_texts[style] = {g:[template(*q,style) for q in qs] for g,(qs,_) in pools.items()}
            texts = [template(Grid.from_dict(r["grid"]),r["verification_position"],r["action"],style) for r in train]
            lengths = [len(t) for t in frozen.tokenizer(texts,truncation=False)["input_ids"]]
            summary["templates"][style] = {"example":texts[0],"train_mean_tokens":sum(lengths)/len(lengths),
                                           "train_max_tokens":max(lengths)}
            if style == "github":
                assert texts == [r["input"] for r in train]
            write_jsonl(a.out/f"train_{style}.jsonl",[{**r,"input":s} for r,s in zip(train,texts,strict=True)])
        def evaluate(runner, name, style):
            result = {}
            tick = time.monotonic()
            unique = list(dict.fromkeys(s for prompts in prompt_texts[style].values() for s in prompts))
            cached = dict(zip(unique, runner.predict(unique,check), strict=True))
            for group,(queries,paths) in pools.items():
                replies = [cached[s] for s in prompt_texts[style][group]]
                predictions = aggregate(replies,paths,groups[group])
                m,rs = stats(groups[group],predictions,oracle,True)
                m["unique_queries"] = len(queries)
                result[group] = m
                write_jsonl(a.out/f"{name}_{group}.jsonl",rs)
                write_jsonl(a.out/f"{name}_{group}_local.jsonl",[
                    {"grid":{"rows":q[0].rows,"cols":q[0].cols,"obstacles":q[0].obstacles,
                             "start":q[0].start,"goal":q[0].goal},
                     "position":q[1],"action":q[2],"prediction":s} for q,s in zip(queries,replies,strict=True)])
            summary["arms"][name] = {"metrics":result,"seconds":time.monotonic()-tick}
            write_json(a.out/"results.json",summary)
            print(name,"done",flush=True)
        for style in TEMPLATES:
            evaluate(frozen,"frozen_"+style,style)
        del frozen
        gc.collect()
        for style in TEMPLATES:
            runner = Seq2Seq(str(a.checkpoint),config)
            torch.manual_seed(42)
            runner.model = get_peft_model(runner.model,LoraConfig(task_type="SEQ_2_SEQ_LM",
                r=8,lora_alpha=16,lora_dropout=0.0,target_modules=["q","v"]))
            rows = read_jsonl(a.out/f"train_{style}.jsonl")
            features = runner.encode_rows(rows)
            loader = torch.utils.data.DataLoader(features,batch_size=4,shuffle=True,
                generator=torch.Generator().manual_seed(42),collate_fn=runner.collate)
            parameters = [p for p in runner.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters,lr=3e-4)
            best, logs, updates = None, [], 0
            for epoch in (1,2):
                runner.model.train()
                epoch_loss = 0.
                optimizer.zero_grad(set_to_none=True)
                for i,batch in enumerate(loader,1):
                    check()
                    loss = runner.loss(batch.to(runner.device),split_verification_loss=True)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite loss")
                    epoch_loss += float(loss.detach())
                    (loss/4).backward()
                    if i%4 == 0:
                        torch.nn.utils.clip_grad_norm_(parameters,1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        updates += 1
                dev = {}
                for group in ("dev","dev_generated"):
                    replies = runner.predict(prompt_texts[style][group],check)
                    dev[group] = stats(groups[group],aggregate(replies,pools[group][1],groups[group]),oracle,False)[0]
                rank = (dev["dev"]["macro_f1"]+dev["dev_generated"]["macro_f1"],
                        dev["dev"]["joint_error_accuracy"],-dev["dev_generated"]["false_reject"])
                logs.append({"epoch":epoch,"loss":epoch_loss/len(loader),"development":dev,"rank":rank})
                print(style,epoch,"updates",updates,"rank",rank,flush=True)
                if best is None or rank > best[0]:
                    best = (rank,epoch,{n:p.detach().cpu().clone() for n,p in runner.model.named_parameters() if p.requires_grad})
            runner.model.load_state_dict(best[2],strict=False)
            runner.model.save_pretrained(a.out/f"adapter_{style}")
            assert updates == 250
            evaluate(runner,"trained_"+style,style)
            summary["arms"]["trained_"+style].update(selection=best[1],history=logs,updates=updates)
            if style in ("github","compact"):
                other = "compact" if style == "github" else "github"
                evaluate(runner,f"trained_{style}_inference_{other}",other)
            write_json(a.out/"results.json",summary)
            del optimizer,runner
            gc.collect()
    write_json(a.out/"completed.json",{"arms":list(summary["arms"])})


if __name__ == "__main__":
    main()
