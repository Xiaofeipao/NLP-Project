"""Exercise the balanced dataset through the V2 model's tokenizer, loss and inference APIs on CPU."""
import argparse
from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch

from ppnl.data import read_jsonl
from ppnl.experiment_freeze import file_hash
from . import g_probes
from .audit import require
from .build import write_object
from .model_io import read_split
from .schema import State
from .single import LABELS
from .smoke import cpu_runner, objective_check


def run(dataset, tokenizer):
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    for name, expected in manifest["sha256"].items():
        require(file_hash(dataset / name)==expected,name)
    require(file_hash(tokenizer)==manifest["tokenizer_sha256"],"frozen tokenizer")
    runner = cpu_runner(tokenizer)
    torch = runner.torch
    rows = read_split(dataset,"train",version=g_probes.VERSION,fingerprint=g_probes.template_hash())
    features = g_probes.prepare_training(runner,dataset)
    tie = next(i for i,r in enumerate(rows) if len(r.get("optimal_actions",[]))>1)
    indices = [tie] + [next(i for i,r in enumerate(rows) if r["mode"]=="v_step" and r["target"]==label)
                      for label in LABELS]
    loss = g_probes.batch_loss(runner,runner.collate([features[i] for i in indices]))
    loss.backward()
    grads = [p.grad for p in runner.model.parameters() if p.grad is not None]
    require(torch.isfinite(loss).item() and all(torch.isfinite(g).all().item() for g in grads)
            and any(torch.count_nonzero(g).item() for g in grads),"finite mixed batch backward")
    require(objective_check(torch),"tied optimal mass")

    states = {r["id"]:State.from_record(r) for r in read_jsonl(dataset / "states.jsonl")}
    s = states[rows[tie]["state_id"]]
    scores = g_probes.generation_policy(runner).scores([(s.grid(),s.current)])
    require(len(scores)==1 and len(scores[0])==4 and all(torch.isfinite(torch.tensor(scores[0]))),"four actions")
    # Compare the actual strings sent to predict with the persisted bytes for every row.
    paths = sorted((dataset / "diagnostics").glob("*.jsonl"))
    paths.extend(dataset / f"{split}.jsonl" for split in ("train","dev"))
    checked = {}
    with patch.object(runner,"predict",side_effect=lambda prompts,check:prompts):
        for path in paths:
            selected = [r for r in read_jsonl(path) if r.get("mode","v_step")=="v_step"]
            observations = [(State.from_record(r["state"]) if "state" in r else states[r["state_id"]],r["candidate"])
                            for r in selected]
            require(g_probes.verify_moves(runner,observations)==[r["input"] for r in selected],path.name)
            checked[path.name] = len(selected)
    v = rows[indices[1]]
    runner.output_limit = max(len(features[i]["labels"]) for i in indices)
    reply = g_probes.verify_last(runner,states[v["state_id"]],v["candidate"])
    require(isinstance(reply,str),"single-action greedy decode")
    return {"model":"random one-layer T5, CPU interface check","train_features":len(features),
            "verifier_loss":runner.model.config.ppnl_verifier_loss,
            "V_target_lengths_including_eos":{rows[i]["target"]:len(features[i]["labels"]) for i in indices[1:]},
            "G_mask_cardinalities_including_zero_for_V":dict(Counter(sum(f["policy_actions"]) for f in features)),
            "finite_mixed_backward":True,"all_five_V_labels_in_mixed_batch":True,
            "tied_optimal_mass_invariant":True,"four_action_scores":True,"single_action_greedy_decode":True,
            "V_prompt_parity_rows_by_file":checked,"prompt_sha256":g_probes.template_hash(),
            "tokenizer_sha256":file_hash(tokenizer),"verified_manifest_file_count":len(manifest["sha256"])}


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--tokenizer",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    report = run(args.data,args.tokenizer)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        write_object(args.out,report)
    print(json.dumps(report,indent=2))
