"""Dev-select a fixed seven-direction sweep; preserve every result and checkpoint."""

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.core import Grid, OfficialOracle, parse_verdict, execute
from ppnl.data import read_jsonl, write_json, write_jsonl
from ppnl.evaluation import plan_metrics, verification_metrics
from ppnl.modeling import Seq2Seq, gpu_budget, action_set_loss
from ppnl.optimization import accumulated_batches
from ppnl.policy import NeuralPolicy, greedy_plans
from ppnl.method_sweep import METHODS, verification_input, extract_verdict, repair_once
from ppnl.verifier_guard import guard_verdict

BASE = Path("outputs/phase2_reproduction/run/train/flan_t5_base/best")
CONFIG = Path("part2/protocol.json")
LEDGER = Path("outputs/phase2_reproduction/gpu_ledger.json")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def train_branch(branch, out, config, check):
    """One additional pass over existing V examples; pilot examples excluded."""
    original = read_jsonl("data/policy_v3/train.jsonl")
    verification = [r for r in original if r["task"] == "verification"]
    excluded = {r["id"] for r in verification[:16]}
    rows = [dict(r) for r in verification if r["id"] not in excluded]
    if branch == "augmentation":
        for r in rows:
            r["input"] = verification_input(Grid.from_dict(r["grid"]), r["candidate"], "structured")
    write_json(
        out / "sample_budget.json",
        {
            "ids": [r["id"] for r in rows],
            "excluded_pilot_ids": sorted(excluded),
            "examples": len(rows),
            "lineage_passes": 3,
            "new_logical_examples": 0,
            "note": "same maps/candidates/labels; prompt augmentation changes representation only; independent branches from the two-pass base",
        },
    )
    runner = Seq2Seq(str(BASE), config)
    torch = runner.torch
    torch.set_num_threads(1)
    runner.model.gradient_checkpointing_enable()
    runner.model.config.use_cache = False
    features = runner.encode_rows(rows)
    loader = torch.utils.data.DataLoader(
        features,
        batch_size=4,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        collate_fn=runner.collate,
    )
    optimizer = torch.optim.AdamW(runner.model.parameters(), lr=1e-4, weight_decay=0.01, fused=True)
    runner.model.train()
    optimizer.zero_grad(set_to_none=True)
    updates = 0
    total = 0
    start = time.perf_counter()
    for _, batch, scale, update in accumulated_batches(loader, 4, "examples"):
        check()
        batch = batch.to(runner.device)
        logits = runner.model(**{k: v for k, v in batch.items() if k != "policy_actions"}).logits
        loss = action_set_loss(
            logits, batch["labels"], runner.action_ids, batch["policy_actions"], "optimal_actions"
        )
        if branch == "multitask":
            # Explicit auxiliary validity classification prevents long diagnosis
            # strings diluting the first-token validity gradient.
            loss = loss + torch.nn.functional.cross_entropy(
                logits[:, 0, :].float(), batch["labels"][:, 0]
            )
        if not torch.isfinite(loss).item():
            raise ValueError("Non-finite loss")
        total += float(loss.detach()) * len(batch["labels"])
        (loss * scale).backward()
        if update:
            torch.nn.utils.clip_grad_norm_(runner.model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            if updates % 25 == 0:
                progress = {
                    "branch": branch,
                    "updates": updates,
                    "seconds": time.perf_counter() - start,
                }
                write_json(out / "progress.json", progress)
                print(progress, flush=True)
    runner.model.save_pretrained(out / "best")
    runner.tokenizer.save_pretrained(out / "best")
    write_json(
        out / "training.json",
        {
            "branch": branch,
            "updates": updates,
            "examples": len(rows),
            "epochs": 1,
            "microbatch": 4,
            "effective_batch": 16,
            "learning_rate": 1e-4,
            "mean_loss": total / len(rows),
            "seconds": time.perf_counter() - start,
            "weights_sha256": sha(out / "best/model.safetensors"),
        },
    )
    del logits, loss, batch, optimizer, runner, features, loader
    gc.collect()
    torch.cuda.empty_cache()


def metric(grids, plans, feedbacks, initial):
    bad = [i for i, t in enumerate(initial) if not t.valid]
    gold = [{"valid": t.valid, "step": t.step, "reason": t.reason} for t in initial]
    return {
        "planning": plan_metrics(grids, plans),
        "repair": plan_metrics([grids[i] for i in bad], [plans[i] for i in bad]) if bad else None,
        "verification": verification_metrics(gold, feedbacks),
        "regressions": sum(
            t.valid and not execute(g, p).valid for g, t, p in zip(grids, initial, plans)
        ),
    }


def evaluate(split, methods, out, config, check):
    import torch

    base = Seq2Seq(str(BASE), config)
    scorer = NeuralPolicy(base, check)
    rows = read_jsonl(f"data/policy_v3/{split}.jsonl")
    if split == "dev":
        candidates = [r for r in rows if r["task"] == "verification"]
        unique = {}
        for r in rows:
            unique.setdefault(
                (
                    Grid.from_dict(r["grid"]).layout_id,
                    tuple(r["grid"]["start"]),
                    tuple(r["grid"]["goal"]),
                ),
                r,
            )
        generated = list(unique.values())
    else:
        candidates = rows
        generated = rows
    pools = {"candidates": (candidates, [r["candidate"] for r in candidates])}
    # OOD maps are paired; do not count a second physical generation batch.
    if split != "test_ood":
        grids = [Grid.from_dict(r["grid"]) for r in generated]
        pools["generated"] = (generated, greedy_plans(grids, scorer, masked=False))
    oracle = OfficialOracle("vendor/ppnl/executor-point-sg.py")
    results = {}
    cache = {}
    for pool, (records, candidates) in pools.items():
        grids = [Grid.from_dict(r["grid"]) for r in records]
        initial = [oracle.check(g, p) for g, p in zip(grids, candidates)]
        results[pool] = {}
        for method in methods:
            key = (pool, method.get("weights", "base"), method["style"])
            if key not in cache:
                verifier = (
                    base
                    if key[1] == "base"
                    else Seq2Seq(str(out.parent / "training" / key[1] / "best"), config)
                )
                prompts = [
                    verification_input(g, p, method["style"]) for g, p in zip(grids, candidates)
                ]
                start = time.perf_counter()
                try:
                    raw = verifier.predict(prompts, check)
                    cache[key] = [extract_verdict(v, method["style"]) for v in raw]
                except ValueError as e:
                    # Reject over-length input; never silently truncate or drop cases.
                    results[pool][method["name"]] = {"not_run": str(e)}
                    if verifier is not base:
                        del verifier
                        gc.collect()
                        torch.cuda.empty_cache()
                    continue
                if verifier is not base:
                    del verifier
                    gc.collect()
                    torch.cuda.empty_cache()
                write_jsonl(
                    out / f"{split}_{pool}_{key[1]}_{key[2]}_verifier.jsonl",
                    [
                        {"id": r["id"], "prediction": v, "raw": text}
                        for r, v, text in zip(records, cache[key], raw)
                    ],
                )
            predictions = cache[key]
            outputs = []
            feedbacks = []
            traces = []
            cost = Counter()
            start = time.perf_counter()
            before = dict(scorer.usage)
            for r, g, candidate, prediction, t in zip(
                records, grids, candidates, predictions, initial
            ):
                check()
                value = candidate
                remaining = 36
                history = []
                first = None
                for round_index in range(max(1, method["rounds"])):
                    if round_index == 0:
                        pred = prediction
                    elif method["guard"] == "executor":
                        pred = ""
                    else:
                        pred = extract_verdict(
                            base.predict([verification_input(g, value, method["style"])], check)[0],
                            method["style"],
                        )
                        cost["extra_verifier_calls"] += 1
                    fb = (
                        execute(g, value).label()
                        if method["guard"] == "executor"
                        else guard_verdict(g, value, pred, method["guard"])
                    )
                    if first is None:
                        first = fb
                    if method["rounds"] == 0:
                        break
                    cost["repair_attempts"] += 1
                    allowance = min(12, remaining) if method["rounds"] > 1 else remaining
                    revised, used, status = repair_once(
                        g, value, fb, scorer, allowance, method["decoder"]
                    )
                    remaining -= used
                    cost["new_actions"] += used
                    history.append(
                        {
                            "feedback": fb,
                            "before": value,
                            "after": revised,
                            "used": used,
                            "status": status,
                        }
                    )
                    unchanged = revised == value
                    value = revised
                    if (
                        status
                        in (
                            "accepted",
                            "success",
                            "malformed_feedback",
                            "bad_step",
                            "illegal_prefix",
                        )
                        or unchanged
                        or remaining <= 0
                    ):
                        break
                oracle.check(g, value)
                outputs.append(value)
                feedbacks.append(first)
                traces.append(
                    {
                        "id": r["id"],
                        "grid": r["grid"],
                        "candidate": candidate,
                        "prediction": prediction,
                        "feedback": first,
                        "plan": value,
                        "history": history,
                        "gold": t.label(),
                    }
                )
            m = metric(grids, outputs, feedbacks, initial)
            m["cost"] = {
                **dict(cost),
                "repair_seconds": time.perf_counter() - start,
                "policy": {k: scorer.usage[k] - before[k] for k in before},
            }
            results[pool][method["name"]] = m
            write_json(out / f"{split}_results.json", results)
            write_jsonl(out / f'{split}_{pool}_{method["name"]}_traces.jsonl', traces)
            print(split, pool, method["name"], m["planning"], flush=True)
    del scorer, base
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    global BASE, CONFIG, LEDGER
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=BASE)
    p.add_argument("--model-config", type=Path, default=CONFIG)
    p.add_argument("--ledger", type=Path, required=True)
    args = p.parse_args()
    BASE = args.checkpoint
    CONFIG = args.model_config
    LEDGER = args.ledger
    out = args.out
    if out.exists():
        raise FileExistsError("Use a new study output")
    out.mkdir(parents=True)
    config = json.loads(CONFIG.read_text())
    config["training"]["eval_batch_size"] = 4
    write_json(
        out / "protocol.json",
        {
            "methods": METHODS,
            "base_sha256": sha(BASE / "model.safetensors"),
            "code_sha256": {
                n: sha(n)
                for n in [
                    __file__,
                    "ppnl/method_sweep.py",
                    "ppnl/verifier_guard.py",
                    "ppnl/modeling.py",
                ]
            },
            "data_sha256": {
                s: sha(f"data/policy_v3/{s}.jsonl") for s in ("train", "dev", "test_id", "test_ood")
            },
            "selection": "dev generated success, invalid candidate repair success, generated optimality, verification joint accuracy; ties retain earlier simpler method",
            "training": "independent one-pass V branches from common two-pass model; exclude first 16 V pilot IDs; no new logical examples",
            "scope": "bounded representatives of seven directions, not every conceivable variant; reused exploratory tests",
            "repair_budget": "36 new actions per example; iterative at most 3 rounds x12; verifier calls are separate costs",
            "tool_trace": "executor observations reveal errors; not unaided CoT or learned diagnosis",
            "primary": "G and R always frozen base; only V weights vary in training branches",
        },
    )
    with gpu_budget(LEDGER, 20, "seven_methods:dev_train_test") as check:
        for branch in ("continuation_control", "multitask", "augmentation"):
            print("TRAIN", branch, flush=True)
            train_branch(branch, out / "training" / branch, config, check)
        devout = out / "evaluation"
        devout.mkdir()
        dev = evaluate("dev", METHODS, devout, config, check)

        def rank(m):
            c = dev["candidates"][m["name"]]
            g = dev["generated"][m["name"]]
            return (
                g["planning"]["success_rate"],
                c["repair"]["success_rate"],
                g["planning"]["optimality"],
                c["verification"]["joint_error_accuracy"],
            )

        eligible = [
            m
            for m in METHODS
            if not m.get("control")
            and "not_run" not in dev["candidates"][m["name"]]
            and "not_run" not in dev["generated"][m["name"]]
        ]
        best = max(eligible, key=rank)
        selection = {
            "method": best,
            "dev_rank": rank(best),
            "dev_sha256": sha(devout / "dev_results.json"),
            "weights_sha256": sha(
                (out / "training" / best["weights"] / "best" if "weights" in best else BASE)
                / "model.safetensors"
            ),
        }
        write_json(out / "selection.json", selection)
        print("SELECTED", selection, flush=True)
        # Freeze once; all predeclared methods retained, no test re-selection.
        evaluate("test_id", METHODS, devout, config, check)
        evaluate("test_ood", METHODS, devout, config, check)
    write_json(
        out / "completed.json",
        {"completed": True, "selected": best, "selection_not_changed_by_test": True},
    )


if __name__ == "__main__":
    main()
