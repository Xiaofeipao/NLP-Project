"""Freeze and evaluate R alternatives on the same invalid cases; no training."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from contextlib import nullcontext
from types import SimpleNamespace

from ppnl.core import Grid, OfficialOracle
from ppnl.data import read_jsonl, write_json, write_jsonl
from ppnl.evaluation import plan_metrics
from ppnl.modeling import Seq2Seq, gpu_budget
from ppnl.policy import NeuralPolicy
from ppnl.repair_variants import VARIANTS, repair


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/phase2_reproduction/run/train/flan_t5_base/best"),
    )
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new output directory")
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    expected = json.loads(Path("part3/best_overall.json").read_text())[
        "generator_checkpoint_sha256"
    ]
    if sha(args.checkpoint / "model.safetensors") != expected:
        raise ValueError("This replay requires the recorded base weights")
    # Frozen generation failures came from this exact same checkpoint.
    old = json.loads(Path("outputs/phase2_feedback/run_001/protocol.json").read_text())
    if old["checkpoint_sha256"] != expected:
        raise ValueError("Generation replay weights differ")
    dev = read_jsonl("data/policy_v3/dev.jsonl")
    cases = {
        "dev_candidates": [
            {"id": r["id"], "grid": r["grid"], "candidate": r["candidate"]}
            for r in dev
            if r["task"] == "verification" and not r["valid"]
        ]
    }
    generated = json.loads((args.checkpoint.parent / "dev_epoch_2.json").read_text())["planning"]
    oracle = OfficialOracle("vendor/ppnl/executor-point-sg.py")
    cases["dev_generated"] = [
        {"id": f"dev-generated-{i}", "grid": r["grid"], "candidate": r["plan"]}
        for i, r in enumerate(generated)
        if not oracle.check(Grid.from_dict(r["grid"]), r["plan"]).valid
    ]
    for split in ("test_id", "test_ood"):
        cases[split] = [
            {"id": r["id"], "grid": r["grid"], "candidate": r["candidate"]}
            for r in read_jsonl(f"data/policy_v3/{split}.jsonl")
            if not r["valid"]
        ]
    cases["test_generated"] = [
        {"id": r["id"], "grid": r["grid"], "candidate": r["candidate"]}
        for r in read_jsonl("outputs/phase2_feedback/run_001/test_id_generated_traces.jsonl")
        if r["method"] == "direct" and r["gold"] != "valid"
    ]
    args.out.mkdir(parents=True)
    write_json(
        args.out / "protocol.json",
        {
            "variants": VARIANTS,
            "weights_sha256": expected,
            "cases_sha256": {
                k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
                for k, v in cases.items()
            },
            "counts": {k: len(v) for k, v in cases.items()},
            "feedback": "same exact executor diagnosis for every R",
            "probabilities": "softmax over four action logits, temperature 1; not calibrated route success probability",
            "selection": "dev mean unconditional optimality first, mean success second, fewer score rows, earlier method; every optimal plan must succeed",
            "device": args.device,
            "code_sha256": {f: sha(f) for f in [__file__, "ppnl/repair_variants.py"]},
            "scope": "reused exploratory error cases; no new training; erasure preserves endpoint and cannot alone rescue failures",
        },
    )
    config = json.loads(Path("part2/protocol.json").read_text())
    config["training"]["eval_batch_size"] = 4
    reports = {}
    started = time.perf_counter()
    context = (
        gpu_budget(args.ledger, 20, "r_variants:errors")
        if args.device == "cuda"
        else nullcontext(lambda: None)
    )
    with context as check:
        if args.device == "cuda":
            runner = Seq2Seq(str(args.checkpoint), config)
        else:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            torch.set_num_threads(4)
            runner = SimpleNamespace(
                model=AutoModelForSeq2SeqLM.from_pretrained(
                    str(args.checkpoint), local_files_only=True
                ).eval(),
                tokenizer=AutoTokenizer.from_pretrained(
                    str(args.checkpoint), local_files_only=True
                ),
                device="cpu",
                config=config,
                input_limit=512,
                autocast=nullcontext,
            )
        scorer = NeuralPolicy(runner, check)
        for split, rows in cases.items():
            reports[split] = {}
            grids = [Grid.from_dict(r["grid"]) for r in rows]
            feedbacks = [oracle.check(g, r["candidate"]).label() for g, r in zip(grids, rows)]
            for spec in VARIANTS:
                outputs = []
                traces = []
                cost = Counter()
                before = dict(scorer.usage)
                start = time.perf_counter()
                for r, g, fb in zip(rows, grids, feedbacks):
                    check()
                    output, status, usage = repair(g, r["candidate"], fb, scorer, spec)
                    verdict = oracle.check(g, output)
                    outputs.append(output)
                    cost["score_rows"] += usage["score_rows"]
                    cost["removed_actions"] += usage["removed_actions"]
                    traces.append(
                        {
                            **r,
                            "feedback": fb,
                            "plan": output,
                            "verdict": verdict.label(),
                            "status": status,
                            "usage": usage,
                        }
                    )
                reports[split][spec["name"]] = {
                    "metrics": plan_metrics(grids, outputs),
                    "cost": dict(cost),
                    "seconds": time.perf_counter() - start,
                    "policy_usage": {k: scorer.usage[k] - before[k] for k in before},
                }
                write_jsonl(args.out / f'{split}_{spec["name"]}_traces.jsonl', traces)
                write_json(args.out / "results.json", reports)
                print(split, spec["name"], reports[split][spec["name"]]["metrics"], flush=True)
            if split == "dev_generated":

                def rank(spec):
                    a = reports["dev_candidates"][spec["name"]]
                    b = reports["dev_generated"][spec["name"]]
                    return (
                        (a["metrics"]["optimality"] + b["metrics"]["optimality"]) / 2,
                        (a["metrics"]["success_rate"] + b["metrics"]["success_rate"]) / 2,
                        -a["cost"]["score_rows"] - b["cost"]["score_rows"],
                    )

                chosen = max(VARIANTS, key=rank)
                write_json(
                    args.out / "selection.json",
                    {"variant": chosen, "rank": rank(chosen), "selection_source": "dev only"},
                )
    write_json(
        args.out / "completed.json",
        {
            "completed": True,
            "chosen": chosen,
            "device": args.device,
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_calls": 0 if args.device == "cpu" else None,
        },
    )


if __name__ == "__main__":
    main()
