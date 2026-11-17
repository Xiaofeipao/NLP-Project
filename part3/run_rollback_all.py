"""Enumerate every possible rollback position with one fixed CPU R backend."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from part3.run_rollback import make_cases
from ppnl.core import Grid, OfficialOracle, shortest_path
from ppnl.data import write_json, write_jsonl
from ppnl.policy import NeuralPolicy
from ppnl.repair_variants import repair
from ppnl.rollback import choose_prefix, offline_diagnosis


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/phase2_reproduction/run/train/flan_t5_base/best"),
    )
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new output directory")
    cases = make_cases()
    maximum = max(len(r["candidate"].split()) for rows in cases.values() for r in rows)
    spec = {"mode": "beam_soft", "score_cap": 108, "erase": True}
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    expected = json.loads(Path("part3/best_overall.json").read_text())[
        "generator_checkpoint_sha256"
    ]
    if sha(args.checkpoint / "model.safetensors") != expected:
        raise ValueError("Saved feedback weights differ")
    args.out.mkdir(parents=True)
    write_json(
        args.out / "protocol.json",
        {
            "k_range": [0, maximum],
            "backend": spec,
            "feedback": "executor",
            "weights_sha256": expected,
            "counts": {s: len(v) for s, v in cases.items()},
            "device": "cpu",
            "selection": "dev mean unconditional optimality, mean success, fewer logical score rows, smaller K",
            "per_case_best": "posthoc reference only; not an adaptive inference model",
            "cases_sha256": {
                s: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
                for s, v in cases.items()
            },
            "code_sha256": {
                f: sha(f) for f in [__file__, "ppnl/rollback.py", "ppnl/repair_variants.py"]
            },
            "scope": "exploratory reused error sets; no training, no new G calls",
        },
    )
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    torch.set_num_threads(4)
    config = json.loads(Path("part2/protocol.json").read_text())
    runner = SimpleNamespace(
        model=AutoModelForSeq2SeqLM.from_pretrained(
            str(args.checkpoint), local_files_only=True
        ).eval(),
        tokenizer=AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True),
        device="cpu",
        config=config,
        input_limit=512,
        autocast=nullcontext,
    )
    scorer = NeuralPolicy(runner)
    oracle = OfficialOracle("vendor/ppnl/executor-point-sg.py")
    reports = {}
    selected = None
    started = time.perf_counter()
    for split, rows in cases.items():
        counters = {
            k: {"success": 0, "feasible": 0, "optimal": 0, "score_rows": 0, "kept_actions": 0}
            for k in range(maximum + 1)
        }
        details = []
        for i, r in enumerate(rows):
            g = Grid.from_dict(r["grid"])
            fb = oracle.check(g, r["candidate"]).label()
            best_len = len(shortest_path(g))
            outcomes = []
            cache = {}
            for k in range(maximum + 1):
                prefix, usage = choose_prefix(g, r["candidate"], fb, k)
                if prefix not in cache:
                    control = f"invalid | step={len(prefix.split())+1} | reason=goal_not_reached"
                    plan, status, telemetry = repair(g, prefix, control, scorer, spec)
                    verdict = oracle.check(g, plan)
                    cache[prefix] = {
                        "plan": plan,
                        "status": status,
                        "score_rows": telemetry["score_rows"],
                        "success": verdict.valid,
                        "feasible": verdict.feasible,
                        "optimal": verdict.valid and len(plan.split()) == best_len,
                    }
                outcome = {"k": k, "keep_actions": usage["keep_actions"], **cache[prefix]}
                outcomes.append(outcome)
                for field in ("success", "feasible", "optimal", "score_rows", "kept_actions"):
                    counters[k][field] += outcome[
                        "keep_actions" if field == "kept_actions" else field
                    ]
            first_opt = next((x["k"] for x in outcomes if x["optimal"]), None)
            best = max(
                outcomes,
                key=lambda x: (x["optimal"], x["success"], -len(x["plan"].split()), -x["k"]),
            )
            details.append(
                {
                    **r,
                    "executor_feedback": fb,
                    "shortest_length": best_len,
                    "diagnosis": offline_diagnosis(g, r["candidate"], r["model_feedback"]),
                    "executor_diagnosis": offline_diagnosis(g, r["candidate"], fb),
                    "first_optimal_k": first_opt,
                    "posthoc_best_k": best["k"],
                    "outcomes": outcomes,
                }
            )
            if (i + 1) % 25 == 0:
                print(split, i + 1, "/", len(rows), flush=True)
        n = len(rows)
        reports[split] = {
            str(k): {
                "n": n,
                "success_rate": v["success"] / n,
                "feasibility": v["feasible"] / n,
                "optimality": v["optimal"] / n,
                "score_rows": v["score_rows"],
                "kept_actions": v["kept_actions"],
            }
            for k, v in counters.items()
        }
        write_jsonl(args.out / f"{split}_all_positions.jsonl", details)
        write_json(args.out / "results.json", reports)
        if split == "dev_generated":

            def rank(k):
                a = reports["dev_candidates"][str(k)]
                b = reports["dev_generated"][str(k)]
                return (
                    (a["optimality"] + b["optimality"]) / 2,
                    (a["success_rate"] + b["success_rate"]) / 2,
                    -a["score_rows"] - b["score_rows"],
                    -k,
                )

            selected = max(range(maximum + 1), key=rank)
            write_json(
                args.out / "selection.json",
                {"k": selected, "rank": rank(selected), "source": "dev only"},
            )
            print("SELECTED_K", selected, rank(selected), flush=True)
        print(split, "done", "selected metrics", reports[split].get(str(selected)), flush=True)
    write_json(
        args.out / "completed.json",
        {
            "completed": True,
            "selected_k": selected,
            "elapsed_seconds": time.perf_counter() - started,
            "device": "cpu",
            "gpu_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
