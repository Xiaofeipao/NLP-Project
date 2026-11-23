"""Generate raw text and score it afterwards. This module contains no grid simulator."""
from collections import Counter, defaultdict
from itertools import combinations
import json

import torch

from objectives import pad_sequences
from texts import ACTIONS, CLASSES, jsonl, parse, prompt, save_json, sha, write_lines


@torch.inference_mode()
def generate_ids(model, encoded, samples=1, sample=False):
    inputs = pad_sequences(encoded, 0, model.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ids = model.generate(input_ids=inputs, attention_mask=inputs.ne(0).long(),
                             max_new_tokens=32, do_sample=sample, num_beams=1, num_return_sequences=samples,
                             **({"top_k":0, "top_p":1.0} if sample else {}), use_cache=True)
    result = []
    for row in ids.cpu().tolist():
        assert row[0] == 0
        seq = row[1:]
        if 1 in seq:
            seq = seq[:seq.index(1)+1]
        result.append(tuple(seq))
    return result


def outputs(model, codec, encoded, batch=12):
    return [codec.decode(ids) for start in range(0, len(encoded), batch)
            for ids in generate_ids(model, encoded[start:start+batch])]


def ratio(n, d):
    return n/d if d else None


def verdict_metrics(gold, raw):
    matrix = defaultdict(Counter)
    exact = Counter()
    invalid = step = reason = joint = binary = formatted = 0
    for g, s in zip(gold, raw, strict=True):
        y, idx = parse(g)
        p, num = parse(s)
        matrix[y][p or "malformed"] += 1
        formatted += p is not None
        binary += p is not None and ((y=="valid")==(p=="valid"))
        exact[y] += int(s==g)
        if y != "valid":
            invalid += 1
            step += num==idx
            reason += p==y
            joint += num==idx and p==y
    classes = {}
    for y, row in matrix.items():
        tp = row.get(y, 0)
        n = sum(row.values())
        predicted = sum(r.get(y, 0) for r in matrix.values())
        classes[y] = dict(n=n, exact=exact[y]/n, precision=ratio(tp, predicted),
                          recall=tp/n, f1=2*tp/(n+predicted))
    binary_counts = Counter()
    for g, s in zip(gold, raw, strict=True):
        y, _ = parse(g)
        p, _ = parse(s)
        binary_counts[("valid" if y=="valid" else "invalid",
                       "malformed" if p is None else ("valid" if p=="valid" else "invalid"))] += 1
    binary_f1 = {}
    for y in ("valid","invalid"):
        actual = sum(n for (a, _), n in binary_counts.items() if a==y)
        if actual:
            predicted = sum(n for (_, b), n in binary_counts.items() if b==y)
            binary_f1[y] = 2*binary_counts[y,y]/(actual+predicted)
    return dict(n=len(raw), format=formatted/len(raw), exact=sum(exact.values())/len(raw),
                binary_accuracy=binary/len(raw), binary_f1_by_observed_class=binary_f1,
                binary_macro_f1=sum(binary_f1.values())/len(binary_f1),
                macro_exact=sum(c["exact"] for c in classes.values())/len(classes),
                class_macro_f1=sum(c["f1"] for c in classes.values())/len(classes),
                invalid_n=invalid, invalid_step=ratio(step,invalid), invalid_reason=ratio(reason,invalid),
                invalid_step_reason=ratio(joint,invalid), classes=classes, confusion=dict(matrix))


def local_eval(model, codec, rows, directory):
    raw = outputs(model, codec, codec.encode_rows(rows))
    saved = [dict(id=r["id"],task=r["task"],gold=r["answers"],raw=s) for r,s in zip(rows,raw,strict=True)]
    write_lines(directory/"dev_raw.jsonl", saved)
    g = [r["raw"] in r["gold"] for r in saved if r["task"]=="G"]
    v = [(r,s) for r,s in zip(rows,raw,strict=True) if r["task"]=="V"]
    metrics = verdict_metrics([r["answers"][0] for r,_ in v], [s for _,s in v])
    groups, histories, lengths = defaultdict(list), defaultdict(list), defaultdict(list)
    for r, s in v:
        pre, rest = r["text"].split("\nPath: [")
        actions = rest.split("]\n",1)[0].split(", ")
        good = r["answers"][0] == s
        groups[r["case"]].append((r["action"],r["label"],good))
        histories[pre,len(actions),actions[-1]].append((r["label"],good))
        lengths[len(actions)].append(good)
    switch, both = Counter(), Counter()
    for grp in groups.values():
        for (a,y,ok), (b,z,other) in combinations(grp,2):
            if a in ACTIONS and b in ACTIONS and y!=z:
                key = "/".join(sorted((y,z)))
                switch[key] += 1
                both[key] += ok and other
    history_pairs = [ok and other for grp in histories.values() for (y,ok),(z,other) in combinations(grp,2) if y!=z]
    return dict(G=dict(n=len(g),optimal_next=sum(g)/len(g)), V=metrics,
                changed_direction={k:dict(n=n,both_exact=both[k]/n) for k,n in switch.items()},
                changed_history=dict(n=len(history_pairs),both_exact=ratio(sum(history_pairs),len(history_pairs))),
                exact_by_path_length={k:dict(n=len(v),accuracy=sum(v)/len(v)) for k,v in lengths.items()})


def plan_eval(model, codec, heldout, split, directory, variants):
    manifest = json.loads((heldout/"manifest.json").read_bytes())
    source, reference = heldout/f"{split}_inputs.jsonl", heldout/f"{split}_gold.jsonl"
    assert sha(source)==manifest["files"][source.name]
    inputs = jsonl(source)
    results = {}
    for field in variants:
        part = [r for r in inputs if r[field] is not None]
        raw = outputs(model, codec, codec.encode_rows([{"text":r[field]} for r in part]))
        write_lines(directory/f"{split}_{field}_raw.jsonl", [dict(id=r["id"],raw=s) for r,s in zip(part,raw,strict=True)])
        assert sha(reference)==manifest["files"][reference.name]
        lookup = {r["id"]:r["gold"] for r in jsonl(reference)}
        results[field] = verdict_metrics([lookup[r["id"]] for r in part], raw)
        lengths=defaultdict(list)
        for row,s in zip(part,raw,strict=True):
            path=row[field].split("\nPath: [")[1].split("]\n")[0]
            lengths[len(path.split(", ")) if path else 0].append(s==lookup[row["id"]])
        results[field]["exact_by_observed_input_actions"]={n:dict(count=len(v),exact=sum(v)/len(v)) for n,v in sorted(lengths.items())}
    return results


def rollout(model, codec, heldout, directory, cap=24):
    manifest = json.loads((heldout/"manifest.json").read_bytes())
    for name in ("rollout_inputs.jsonl","rollout_gold.jsonl"):
        assert sha(heldout/name)==manifest["files"][name]
    tasks = jsonl(heldout/"rollout_inputs.jsonl")
    records = {r["id"]:dict(id=r["id"],path=[],verdicts=[],finish=None) for r in tasks}
    for _ in range(cap):
        live = [r for r in tasks if records[r["id"]]["finish"] is None]
        if not live:
            break
        def infer(rows, task):
            return outputs(model,codec,codec.encode_rows([dict(text=prompt(r["grid"],records[r["id"]]["path"],task)) for r in rows]))
        for r, action in zip(live,infer(live,"G"),strict=True):
            row = records[r["id"]]
            row["path"].append(action)
            if action not in ACTIONS:
                row["finish"]="malformed_G"
        candidates = [r for r in live if records[r["id"]]["finish"] is None]
        for r, verdict in zip(candidates,infer(candidates,"V"),strict=True):
            row = records[r["id"]]
            row["verdicts"].append(verdict)
            cls, _ = parse(verdict)
            if cls!="goal_not_reached":
                row["finish"]=cls or "malformed_V"
    for row in records.values():
        if row["finish"] is None:
            row["finish"]="cap"
    write_lines(directory/"dev_rollout_raw.jsonl",records.values())
    gold = {r["id"]:r["paths"] for r in jsonl(heldout/"rollout_gold.jsonl")}
    success = sum(r["finish"]=="valid" and r["path"] in gold[r["id"]] for r in records.values())
    return dict(n=len(records),cap=cap,stop=dict(Counter(r["finish"] for r in records.values())),
                generated_valid_and_whole_path_shortest=success,shortest_success=success/len(records))


def history_contrast(model, codec, heldout, directory):
    manifest=json.loads((heldout/"manifest.json").read_bytes())
    for name in ("contrast_inputs.jsonl","contrast_gold.jsonl"):
        assert sha(heldout/name)==manifest["files"][name]
    rows=jsonl(heldout/"contrast_inputs.jsonl")
    raw=outputs(model,codec,codec.encode_rows(rows))
    predictions={r["id"]:s for r,s in zip(rows,raw,strict=True)}
    write_lines(directory/"dev_history_contrast_raw.jsonl",[dict(id=k,raw=v) for k,v in predictions.items()])
    pairs=jsonl(heldout/"contrast_gold.jsonl")
    good=sum(all(predictions[i]==g for i,g in zip(pair["ids"],pair["golds"],strict=True)) for pair in pairs)
    changed=sum(parse(predictions[p["ids"][0]])[0]!=parse(predictions[p["ids"][1]])[0] and
                all(parse(predictions[i])[0] is not None for i in p["ids"]) for p in pairs)
    return dict(n=len(pairs),unique_prompts=len(rows),both_exact=good/len(pairs),generated_class_changes=changed/len(pairs))
