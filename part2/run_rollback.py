"""Original Part2 greedy R: only the reported-prefix rollback distance varies."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

from part3.run_rollback import make_cases
from ppnl.core import Grid, execute, parse_verdict, OfficialOracle, shortest_path
from ppnl.data import write_json, write_jsonl
from ppnl.policy import NeuralPolicy
from ppnl.repair_variants import repair


def reported_prefix(grid, candidate, feedback, k):
    q = parse_verdict(feedback)
    if q is None or q[0]:
        return None, "malformed_feedback" if q is None else "accepted"
    _, step, reason = q
    actions = candidate.split()
    if (
        reason == "goal_not_reached"
        and step != len(actions) + 1
        or reason != "goal_not_reached"
        and step > len(actions)
    ):
        return None, "bad_step"
    prefix = " ".join(actions[: max(0, step - 1 - k)])
    if not execute(grid, prefix).feasible:
        return None, "illegal_prefix"
    return prefix, "ready"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new output directory")
    checkpoint = Path("outputs/phase2_reproduction/run/train/flan_t5_base/best")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    expected = json.loads(Path("part3/best_overall.json").read_text())[
        "generator_checkpoint_sha256"
    ]
    if sha(checkpoint / "model.safetensors") != expected:
        raise ValueError("Weights differ from feedback replay")
    cases = make_cases()
    maximum = max(len(r["candidate"].split()) for rows in cases.values() for r in rows)
    args.out.mkdir(parents=True)
    write_json(
        args.out / "protocol.json",
        {
            "r": "original raw greedy continuation; no masks, penalties, branching or erasure",
            "k_range": [0, maximum],
            "max_new_actions": 36,
            "device": "cpu",
            "weights_sha256": expected,
            "sources": ["model", "executor"],
            "counts": {s: len(r) for s, r in cases.items()},
            "selection": "dev mean unconditional optimality then mean success then smaller K, separately per feedback source",
            "feedback_sanity": "same as original Part2; invalid report indices rejected, illegal retained prefixes rejected; no true-error correction",
            "code_sha256": {f: sha(f) for f in [__file__, "ppnl/repair_variants.py"]},
            "status": "reused exploratory error cases; no training, no new G or V inference",
        },
    )
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(4)
    config = json.loads(Path("part2/protocol.json").read_text())
    runner = SimpleNamespace(
        model=AutoModelForSeq2SeqLM.from_pretrained(str(checkpoint), local_files_only=True).eval(),
        tokenizer=AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True),
        device="cpu",
        config=config,
        input_limit=512,
        autocast=nullcontext,
    )
    scorer = NeuralPolicy(runner)
    oracle = OfficialOracle("vendor/ppnl/executor-point-sg.py")
    reports = {}
    selected = {}
    started = time.perf_counter()
    for split, rows in cases.items():
        reports[split] = {}
        for source in ["model", "executor"]:
            counts = {k: {"success": 0, "optimal": 0, "feasible": 0} for k in range(maximum + 1)}
            details = []
            for r in rows:
                g = Grid.from_dict(r["grid"])
                true = oracle.check(g, r["candidate"]).label()
                feedback = r["model_feedback"] if source == "model" else true
                bestlen = len(shortest_path(g))
                outcomes = []
                cache = {}
                for k in range(maximum + 1):
                    prefix, status = reported_prefix(g, r["candidate"], feedback, k)
                    if prefix is None:
                        plan = r["candidate"]
                    else:
                        if prefix not in cache:
                            fb = f"invalid | step={len(prefix.split())+1} | reason=goal_not_reached"
                            cache[prefix] = repair(g, prefix, fb, scorer, {"mode": "raw"})[:2]
                        plan, status = cache[prefix]
                    verdict = oracle.check(g, plan)
                    opt = verdict.valid and len(plan.split()) == bestlen
                    counts[k]["success"] += verdict.valid
                    counts[k]["feasible"] += verdict.feasible
                    counts[k]["optimal"] += opt
                    outcomes.append(
                        {
                            "k": k,
                            "plan": plan,
                            "status": status,
                            "success": verdict.valid,
                            "feasible": verdict.feasible,
                            "optimal": opt,
                        }
                    )
                details.append(
                    {
                        **r,
                        "source_feedback": feedback,
                        "gold": true,
                        "shortest_length": bestlen,
                        "outcomes": outcomes,
                    }
                )
            n = len(rows)
            reports[split][source] = {
                str(k): {
                    "n": n,
                    "success_rate": v["success"] / n,
                    "feasibility": v["feasible"] / n,
                    "optimality": v["optimal"] / n,
                }
                for k, v in counts.items()
            }
            write_jsonl(args.out / f"{split}_{source}_all_positions.jsonl", details)
            write_json(args.out / "results.json", reports)
            print(
                split,
                source,
                "K0",
                reports[split][source]["0"],
                "K36",
                reports[split][source][str(maximum)],
                flush=True,
            )
        if split == "dev_generated":
            for source in ["model", "executor"]:

                def rank(k):
                    a = reports["dev_candidates"][source][str(k)]
                    b = reports["dev_generated"][source][str(k)]
                    return (
                        (a["optimality"] + b["optimality"]) / 2,
                        (a["success_rate"] + b["success_rate"]) / 2,
                        -k,
                    )

                best = max(range(maximum + 1), key=rank)
                baseline_success = rank(0)[1]
                safe = max(
                    [k for k in range(maximum + 1) if rank(k)[1] >= baseline_success], key=rank
                )
                selected[source] = {
                    "optimality_first_k": best,
                    "rank": rank(best),
                    "no_dev_success_drop_k": safe,
                    "safe_rank": rank(safe),
                }
            write_json(args.out / "selection.json", selected)
            print("SELECTED", selected, flush=True)
    write_json(
        args.out / "completed.json",
        {
            "completed": True,
            "selection": selected,
            "elapsed_seconds": time.perf_counter() - started,
            "device": "cpu",
            "gpu_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
