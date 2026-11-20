"""Convert the mixed G/V dataset to one five-label verification task."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import random
from shutil import copyfile
from tempfile import TemporaryDirectory

from ppnl.core import ACTIONS, Grid, OfficialOracle, execute
from ppnl.data import read_jsonl
from ppnl.experiment_freeze import file_hash
from ppnl.policy import neighbors
from . import single
from .audit import check_g, require
from .build import SOURCE_SHA, V2_COMMIT, write_lines, write_object
from .model_io import read_split
from .schema import State, render


def quotas(split):
    counts = (1000, 250, 250, 250, 250) if split == "train" else (50, 13, 13, 12, 12)
    return dict(zip(single.LABELS, counts, strict=True))


def traces(parents, registry, raw=False):
    buckets = defaultdict(list)
    for r in parents:
        if raw:
            if "candidate" not in r:
                continue
            grid, donor = Grid.from_dict(r["grid"]), r["id"]
        else:
            grid, donor = registry[r["state_id"]].grid(), r["donor_id"]
        for tick in execute(grid, r["candidate"]).trace:
            if tick["reason"] == "goal_not_reached":
                continue
            s, a = State.from_grid(grid, tick["from"]), tick["action"]
            entry = (s, a, donor, {"kind": "original_path" if raw else "mixed_path",
                                    "parent": r["id"], "step": tick["step"]})
            buckets[tick["reason"] or "valid"].append(entry)
    return buckets


def convert(split, old, raw, registry, rng):
    g_rows = [dict(r) for r in old if r["mode"] == "g_next"]
    g_by_state = {r["state_id"]: r for r in g_rows}
    plans = [r for r in old if r["mode"] == "v_plan"]
    local = [r for r in old if r["mode"] == "v_step"]
    old_shared = {r["state_id"] for r in plans+local} & g_by_state.keys()
    used_states = {r["state_id"]: registry[r["state_id"]] for r in g_rows}
    source_layouts = {r["layout_id"] for r in raw}
    covered = {s.record()["layout_id"] for s in used_states.values()}
    rows, seen, vstates, labels = [], set(), set(), Counter()
    targets = quotas(split)

    def add(state, action, donor, origin, keep_id=None):
        key = (state.id, action)
        if key in seen:
            return False
        label = single.reference_label(state, action)
        if labels[label] >= targets[label]:
            return False
        rows.append(dict(id=keep_id or f"{split}-single-{len(rows)+1:04}", mode="v_step", task="verification",
                         state_id=state.id, candidate=action, input=single.step_input(state, action),
                         target=label, donor_id=donor, source=origin))
        seen.add(key)
        used_states[state.id] = state
        covered.add(state.record()["layout_id"])
        vstates.add(state.id)
        labels[label] += 1
        return True

    for r in local:
        require(add(registry[r["state_id"]], r["candidate"] or None, r["donor_id"],
                    {"kind": "mixed_step", "parent": r["id"], "pair": r["group_id"]}, r["id"]), r["id"])

    # Pair STOP away with STOP exactly at the same position by changing only the goal.
    gs = g_rows.copy()
    rng.shuffle(gs)
    gs.sort(key=lambda g: g["state_id"] not in old_shared - vstates)
    new_stops = []
    for g in gs:
        if labels["goal_not_reached"] == targets["goal_not_reached"]:
            break
        s = registry[g["state_id"]]
        at = replace(s, goal=s.current)
        if (s.id,None) in seen or (at.id,None) in seen:
            continue
        origin = {"kind": "stop_pair", "parent": g["id"]}
        require(add(s,None,g["donor_id"],origin) and add(at,None,g["donor_id"],origin), g["id"])
        new_stops.append((at,g))

    # A move after an intermediate arrival is still judged for legality.
    for s,g in new_stops[:20 if split == "train" else 2]:
        grid, _ = s.witness()
        a = rng.choice(list(neighbors(grid,s.current)))
        require(add(s,a,g["donor_id"],{"kind": "depart_goal", "parent": g["id"]}), g["id"])

    positive = [r for r in plans if r["target"] == "valid"]
    rng.shuffle(positive)
    for p in positive:
        s = registry[p["state_id"]]
        if (s.id in old_shared and s.id not in vstates) or s.record()["layout_id"] not in covered:
            add(s,p["candidate"].split()[0],p["donor_id"],{"kind": "mixed_path","parent":p["id"],"step":1})
    require(old_shared <= vstates and covered == source_layouts, f"{split} lost shared states or layouts")

    # Prefer genuine prefix positions and first failures from the candidate paths.
    for buckets in (traces(plans,registry), traces(raw,registry,raw=True)):
        for label, entries in buckets.items():
            rng.shuffle(entries)
            for observation in entries:
                if labels[label] >= targets[label]:
                    break
                add(*observation)
    require(labels == targets, (split,labels))
    ids_per_state = defaultdict(list)
    for r in rows:
        ids_per_state[r["state_id"]].append(r["id"])
        r["shared_g_id"] = g_by_state[r["state_id"]]["id"] if r["state_id"] in g_by_state else None
    for g in g_rows:
        matches = ids_per_state[g["state_id"]]
        g["anchor_v_id"] = g["anchor_v_id"] if g["anchor_v_id"] in matches else next(iter(matches),None)
    all_rows = g_rows + rows
    rng.shuffle(all_rows)
    return all_rows, used_states


def audit_split(rows, states, old, raw, oracle, split, states_for_source):
    originals = {r["id"]:r for r in old}
    donors = {r["id"]:r for r in raw}
    counts, stop = Counter(), Counter()
    gs = {r["state_id"]:r for r in rows if r["mode"] == "g_next"}
    vs = {r["id"]:r for r in rows if r["mode"] == "v_step"}
    require(len(rows) == len({r["id"] for r in rows}) == len({r["input"] for r in rows}) and
            len(gs) == len(vs) == sum(quotas(split).values()) == len(rows)//2, split)
    pair_members = defaultdict(list)
    for r in rows:
        s = states[r["state_id"]]
        require(donors[r["donor_id"]]["layout_id"] == s.record()["layout_id"],r["id"])
        if r["mode"] == "g_next":
            parent = originals[r["id"]]
            require(all(r[k] == parent[k] for k in ("input","target","optimal_actions","distance","state_id"))
                    and r["input"] == render("g_next",s), r["id"])
            check_g(r,s,oracle)
            if r["anchor_v_id"]:
                require(vs[r["anchor_v_id"]]["state_id"] == s.id,r["id"])
            continue
        a = r["candidate"]
        grid, prefix = s.witness()
        verdict = oracle.check(grid," ".join(prefix + ([] if a is None else [a])))
        valid = verdict.valid if a is None else verdict.feasible
        expected = "valid" if valid else verdict.reason
        require(r["mode"] == "v_step" and r["target"] == expected and r["input"] == single.step_input(s,a),r["id"])
        require(r["shared_g_id"] == (gs[s.id]["id"] if s.id in gs else None), r["id"])
        counts[expected] += 1
        if a is None:
            stop[expected] += 1
        origin = r["source"]
        if origin["kind"] in ("original_path","mixed_path"):
            parent = donors[origin["parent"]] if origin["kind"] == "original_path" else originals[origin["parent"]]
            grid = Grid.from_dict(parent["grid"]) if "grid" in parent else states_for_source[parent["state_id"]].grid()
            step = origin["step"]
            tokens = parent["candidate"].split()
            prior = oracle.check(grid," ".join(tokens[:step-1]))
            position = grid.start if step == 1 else prior.trace[step-2]["to"]
            require(prior.feasible and tokens[step-1] == a and s == State.from_grid(grid,position),r["id"])
        elif origin["kind"] == "stop_pair":
            pair_members[origin["parent"]].append((s,expected))
        elif origin["kind"] == "depart_goal":
            require(s.current == s.goal and a in ACTIONS and expected == "valid",r["id"])
        elif origin["kind"] == "mixed_step":
            p = originals[origin["parent"]]
            require(s.id == p["state_id"] and a == (p["candidate"] or None),r["id"])
    for group in pair_members.values():
        require(len(group)==2 and {x[1] for x in group} == {"valid","goal_not_reached"} and
                replace(group[0][0],goal=group[1][0].goal) == group[1][0], "STOP pair")
    layouts = {s.record()["layout_id"] for s in states.values()}
    require(layouts == {r["layout_id"] for r in raw} and counts == quotas(split)
            and stop["valid"] == stop["goal_not_reached"], split)
    shared_v = [r for r in vs.values() if r["shared_g_id"]]
    return {"G":len(gs), "V":len(vs), "V_labels":dict(counts), "STOP_labels":dict(stop), "layouts":len(layouts),
            "shared_G_states":len({r["state_id"] for r in shared_v}), "shared_V_rows":len(shared_v),
            "G_with_multiple_optimal_directions":sum(len(r["optimal_actions"])>1 for r in gs.values()),
            "V_sources":dict(Counter(r["source"]["kind"] for r in vs.values()))}


def build(source, mixed, executor, output):
    if output.exists():
        raise FileExistsError(output)
    oracle = OfficialOracle(executor)
    require(all(file_hash(source / f)==h for f,h in SOURCE_SHA.items()), "original v2 inputs")
    require(oracle.sha256 == json.loads((source/"manifest.json").read_bytes())["executor_sha256"],"executor")
    mixed_manifest = json.loads((mixed/"manifest.json").read_bytes())
    require(file_hash(mixed/"states.jsonl")==mixed_manifest["sha256"]["states.jsonl"], "mixed states")
    registry = {r["id"]:State.from_record(r) for r in read_jsonl(mixed/"states.jsonl")}
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp)/"dataset"
        stage.mkdir()
        states, layouts, report = {}, {}, {}
        for split,seed in (("train",42),("dev",43)):
            old, raw = read_split(mixed,split), read_jsonl(source/f"{split}.jsonl")
            rows, used = convert(split,old,raw,registry,random.Random(seed))
            write_lines(stage/f"{split}.jsonl",rows)
            report[split] = audit_split(read_jsonl(stage/f"{split}.jsonl"),used,old,raw,oracle,split,registry)
            states.update(used)
            layouts[split] = {s.record()["layout_id"] for s in used.values()}
        for split in ("test_id","test_ood"):
            path = stage/f"{split}.jsonl"
            copyfile(source/path.name,path)
            raw = read_jsonl(path)
            grids = [Grid.from_dict(r["grid"]) for r in raw]
            candidates = [r["candidate"] for r in raw]
            predictions = single.scan(grids,candidates,lambda obs: [single.reference_label(s,a) for s,a in obs])
            gold = [oracle.check(g,p).label() for g,p in zip(grids,candidates,strict=True)]
            require(predictions == gold == [r["label"] for r in raw] and len(raw)==200,split)
            report[split] = {"frozen_candidates":len(raw),"reference_scan_exact":len(raw)}
            layouts[split] = {g.layout_id for g in grids}
        require(layouts["train"].isdisjoint(layouts["dev"] | layouts["test_id"] | layouts["test_ood"])
                and layouts["dev"].isdisjoint(layouts["test_id"] | layouts["test_ood"]), "split overlap")
        write_lines(stage/"states.jsonl",(s.record() for k,s in sorted(states.items())))
        write_object(stage/"audit.json",report)
        write_object(stage/"manifest.json",{"schema":single.VERSION,"template_sha256":single.template_hash(),
            "base_commit":V2_COMMIT,"executor_sha256":oracle.sha256,"train_seed":42,"dev_seed":43,
            "parent_manifest_sha256":file_hash(mixed/"manifest.json"), "source_sha256":SOURCE_SHA,
            "sha256":{p.name:file_hash(p) for p in sorted(stage.iterdir())}})
        stage.rename(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("data/derived"))
    parser.add_argument("--mixed",type=Path,default=Path("data/step_semantic_v3"))
    parser.add_argument("--executor",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source,args.mixed,args.executor,args.out),indent=2))
