"""CPU checks on the desktop. The queued job does not reserve GPU memory."""
from contextlib import nullcontext
from hashlib import file_digest
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def check(root):
    import torch
    from transformers import AutoTokenizer
    work = root / "workspace"
    sys.path.insert(0, str(work))
    from ppnl.core import ACTIONS
    from ppnl.data import read_jsonl
    from step_data_v3 import g_probes
    from step_data_v3.coordinate_paths import DELTAS
    from step_data_v3.evaluate_coordinates import lookup_both
    from step_data_v3.first_token import example_losses
    from step_data_v3.schema import State
    from step_data_v3.single import LABELS
    from step_data_v3.train_shared_cases import case_supervision

    spec = read(root/"spec.json")
    base = root.parent/spec["base_directory"]
    assert all(sha(root/n)==h for n,h in read(root/"manifest.json").items())
    assert sha(base/"model.safetensors")==spec["base_sha256"] and sha(base/"tokenizer.json")==spec["tokenizer_sha256"]
    assert (base/"upstream_revision.txt").read_text().strip()==spec["base_revision"]
    assert DELTAS==dict(ACTIONS)
    data = work/"data/step_g_shared_4000"
    assert sha(data/"manifest.json")==spec["training_manifest_sha256"]
    assert g_probes.template_hash()==spec["checkpoint_prompt_sha256"]
    assert all(sha(data/n)==h for n,h in read(data/"manifest.json")["sha256"].items())
    cases = read_jsonl(data/"train.jsonl")
    rows,bundles,weights,totals=case_supervision(cases)
    assert len(bundles)==4000 and len(rows)==len(weights)==20400 and all(len(b) in (5,6) for b in bundles)
    assert round(totals["G"],5)==2000 and all(round(totals[k],5)==400 for k in LABELS)
    assert not (work/"data/random_errors_20261123").exists()
    assert spec["V_fields"]==["Current","Goal","Walls","Action"]
    for row in rows:
        if row["mode"]=="v_step":
            assert [l.partition(":")[0] for l in row["input"].splitlines()]==spec["V_fields"]
            assert row["input"].endswith("Action: "+row["candidate"])
    annotated=next(r for r in rows if r["id"]=="b2592a0e85d58d496c4f4b9f-V-down")
    assert annotated["input"]=="Current: (1,3)\nGoal: (3,3)\nWalls: [(2,3), (5,4)]\nAction: down"

    tok=AutoTokenizer.from_pretrained(str(base),local_files_only=True)
    verdicts=[tok(k,add_special_tokens=False).input_ids[0] for k in LABELS]
    assert dict(zip(LABELS,verdicts,strict=True))==read(data/"audit.json")["verifier_first_tokens"]
    moves=[tok(a,add_special_tokens=False).input_ids[0] for a in ACTIONS]
    reps=read(root/"smoke_rows.json")
    ids=tok([r["input"] for r in reps],truncation=False).input_ids
    assert all(len(seq)<=spec["input_limit"] and tok.unk_token_id not in seq for seq in ids)
    y=torch.tensor([tok(r["target"],add_special_tokens=False).input_ids[0] for r in reps])
    allowed=torch.tensor([[a in r.get("optimal_actions",[]) for a in ACTIONS] for r in reps])
    logits=torch.zeros((len(reps),len(tok)),requires_grad=True)
    losses=example_losses(logits,y,allowed,moves,verdicts)
    losses.mean().backward()
    assert torch.isfinite(losses).all() and torch.isfinite(logits.grad).all() and (logits.grad!=0).any(1).all()

    class FixedScores:
        device="cpu"
        input_limit=512
        config={"training":{"eval_batch_size":16}}
        tokenizer=tok
        verdict_ids=verdicts
        autocast=staticmethod(nullcontext)

        def first_logits(self,batch):
            scores=torch.zeros((len(batch.input_ids),len(tok)))
            scores[:,verdicts[LABELS.index("valid")]]=3
            scores[:,verdicts[LABELS.index("obstacle_collision")]]=2
            return scores

    state=State(6,6,((1,0),),(0,0),(0,1))
    actions=("up","down","right","wait")
    unique={(state.id,a):(state,a) for a in actions}
    seen,_=lookup_both(FixedScores(),unique,g_probes.step_input)
    for a in actions:
        assert seen[state.id,a]==("valid","valid" if a=="right" else "obstacle_collision")
    return dict(cpu_preflight=True, training_states=len(cases), training_views=len(rows), five_token_ids=verdicts,
                G_weight=totals["G"],each_V_class_weight=400, base_sha256=spec["base_sha256"],
                checked_template=g_probes.template_hash(), V_fields=spec["V_fields"],
                verified_V_training_inputs=16400, coordinate_boundary_and_guard_smoke=True)


if __name__=="__main__":
    root=Path(__file__).resolve().parent
    (root/"cpu_preflight.json").write_text(json.dumps(check(root),indent=2)+"\n",encoding="utf-8")
