"""Build the frozen four-direction training variant from data/step_g_balanced."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from ppnl.core import ACTIONS, OfficialOracle
from ppnl.data import read_jsonl
from ppnl.experiment_freeze import file_hash
from . import g_probes, last_move
from .audit import require
from .audit_paired_actions import coverage, leakage_report, verify_replacements
from .balance_steps import audit_split, token_check
from .build import SOURCE_SHA, write_lines, write_object
from .model_io import read_split
from .paired_actions import control_groups, make_rows, row_key, select_keys
from .schema import State
from .shortcut_audit import inspect_dataset


PARENT_MANIFEST = "aa10f89318ee14b061158a7eb333054c19786adf62af8706813df8871837adbc"


def write_v_views(folder, split, previous, current, states, oracle):
    parents = defaultdict(list)
    old, present = {}, {row_key(r) for r in current if r["mode"]=="v_step"}
    gs = {r["state_id"]:r["id"] for r in current if r["mode"]=="g_next"}
    for r in previous:
        if r["mode"]=="v_step":
            old[row_key(r)] = r["id"]
            parents[r["state_id"]].append(r["id"])
    rows, by_key = [], {}
    for s in sorted(parents):
        for a in ACTIONS:
            key = s,a
            row = dict(g_probes.example(states[s],a),id=f"{split}-v-probe-{s}-{a}",
                       source_v_ids=sorted(parents[s]),source_g_id=gs.get(s),
                       in_parent_supervised_V=key in old,in_same_split_supervised_V=key in present)
            require(row["target"]==g_probes.reference(states[s],a,oracle),row["id"])
            rows.append(row)
            by_key[key] = row["id"]
    require(len(rows)==4*len(parents)==len({r["input"] for r in rows}),split)
    write_lines(folder/f"v_{split}_four.jsonl",rows)
    return {"previous_V_states":len(parents),"four_independent_moves":len(rows),
            "labels":Counter(r["target"] for r in rows),"overlap_with_G_four_moves":4*len(gs.keys()&parents.keys()),
            "also_in_formal_V":sum(r["in_same_split_supervised_V"] for r in rows)},by_key


def build(parent, previous, source, executor, tokenizer, output):
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    require(file_hash(parent/"manifest.json")==PARENT_MANIFEST,"Expected the frozen balanced parent")
    manifest = json.loads((parent/"manifest.json").read_bytes())
    require(all(file_hash(parent/k)==v for k,v in manifest["sha256"].items()),"Changed parent dataset")
    require(all(file_hash(source/k)==v for k,v in SOURCE_SHA.items()),"Unexpected source partition")
    require(file_hash(previous/"manifest.json")==manifest["previous_manifest_sha256"],"Unexpected predecessor")
    oracle = OfficialOracle(executor)
    require(oracle.sha256==manifest["executor_sha256"] and file_hash(tokenizer)==manifest["tokenizer_sha256"],
            "Executor or tokenizer fingerprint mismatch")
    states = {r["id"]:State.from_record(r) for r in read_jsonl(parent/"states.jsonl")}
    old = read_jsonl(parent/"train.jsonl")
    old_vs = [r for r in old if r["mode"]=="v_step"]
    parent_groups = json.loads((parent/"control_groups.json").read_bytes())
    # Selection sees only training rows, their states and the original training controls.
    train_states = {r["state_id"]:states[r["state_id"]] for r in old}
    chosen = select_keys(old,train_states,parent_groups["train"])
    new = make_rows(old,chosen,train_states)
    groups = control_groups(new,parent_groups["train"],train_states)
    anchors = verify_replacements(old,new)
    before, after = (coverage(rows,train_states,old_vs) for rows in (old,new))
    require(after["full_four_states"]>before["full_four_states"] and
            after["full_four_G_states"]>=before["full_four_G_states"] and
            after["paired_G_states"]==187 and after["full_four_previous_V_only_states"]>=18 and
            after["full_four_with_at_least_three_outcomes"]>before["full_four_with_at_least_three_outcomes"],
            "Expected increased directional and geometric coverage")
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp)/"dataset"
        diag = stage/"diagnostics"
        diag.mkdir(parents=True)
        for name in ("dev.jsonl","states.jsonl","test_id.jsonl","test_ood.jsonl"):
            copyfile(parent/name,stage/name)
        write_lines(stage/"train.jsonl",new)
        new = read_jsonl(stage/"train.jsonl")
        write_object(stage/"control_groups.json",{"train":groups,"dev":parent_groups["dev"]})
        dev = read_jsonl(stage/"dev.jsonl")
        tests = {split:read_jsonl(stage/f"{split}.jsonl") for split in ("test_id","test_ood")}
        report = json.loads((parent/"audit.json").read_bytes())
        prior = read_split(previous,"train",version=last_move.VERSION,fingerprint=last_move.template_hash())
        old_axes = report["supervised"]["train"]["previous_shortest_arrival_axes"]
        report["supervised"]["train"] = audit_split(new,groups,{r["state_id"]:states[r["state_id"]] for r in new},
                               prior,read_jsonl(source/"train.jsonl"),oracle,"train",
                               parent_verification={r["id"]:r for r in old_vs})
        report["supervised"]["train"]["previous_shortest_arrival_axes"] = old_axes
        v_views = {}
        for split,pre,now,seed in (("train",old,new,42),("dev",dev,dev,43)):
            report["g_diagnostics"][split] = g_probes.write_g_views(diag,split,now,states,oracle,seed)
            v_views[split], view_ids = write_v_views(diag,split,pre,now,states,oracle)
            if split=="train":
                # Keep every original counterfactual reproducible even if only part of its
                # group fits in the current supervised quota. These are training-source probes.
                old_ids = {r["id"]:row_key(r) for r in old_vs}
                control_view = [dict(g,probe_ids=[view_ids[old_ids[i]] for i in g["rows"]],
                                     supervised_members=sum(old_ids[i] in chosen for i in g["rows"]))
                                for g in parent_groups["train"]]
                write_object(diag/"previous_train_controls.json",control_view)
        for i,(split,rows) in enumerate(tests.items()):
            g_probes.write_heldout_view(diag,split,rows,oracle,44+i)
        frozen = ("dev.jsonl","test_id.jsonl","test_ood.jsonl","states.jsonl","diagnostics/g_dev_four.jsonl",
                  "diagnostics/g_dev_balanced.jsonl","diagnostics/test_id_local.jsonl","diagnostics/test_ood_local.jsonl")
        require(all(file_hash(stage/k)==manifest["sha256"][k] for k in frozen),"Changed comparison partition")
        report["paired_actions"] = {"parent_manifest_sha256":PARENT_MANIFEST,"before":before,"after":after,
                                    "G_anchor_metadata_updates":anchors,"frozen_parent_files":list(frozen),
                                    "V_source_four_move_views":v_views,"leakage":leakage_report(new,dev,tests,states)}
        report["tokenization"] = token_check(stage,tokenizer,states)
        require(all(v["max_input_tokens"]<=128 and v["max_target_tokens"]<=8
                    for v in report["tokenization"]["files"].values()),"Exceeded active trainer input/target limits")
        write_object(stage/"audit.json",report)
        write_object(stage/"shortcut_audit.json",inspect_dataset(stage,True))
        revised = dict(manifest,data_revision="paired-actions-v1",previous_manifest_sha256=PARENT_MANIFEST,
                       sha256={p.relative_to(stage).as_posix():file_hash(p) for p in sorted(stage.rglob("*")) if p.is_file()})
        write_object(stage/"manifest.json",revised)
        stage.rename(output)
    return report["paired_actions"]


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",type=Path,default=Path("data/step_g_balanced"))
    parser.add_argument("--previous",type=Path,default=Path("data/step_last_move"))
    parser.add_argument("--source",type=Path,default=Path("data/derived"))
    parser.add_argument("--executor",type=Path,required=True)
    parser.add_argument("--tokenizer",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.parent,args.previous,args.source,args.executor,args.tokenizer,args.out)["after"],indent=2))
