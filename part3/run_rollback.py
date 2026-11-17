"""CPU rollback comparison with fixed feedback, checkpoint and continuation budgets."""

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from ppnl.core import Grid, OfficialOracle, parse_verdict
from ppnl.data import read_jsonl, write_json, write_jsonl
from ppnl.evaluation import plan_metrics
from ppnl.policy import NeuralPolicy
from ppnl.repair_variants import repair, repaired_prefix
from ppnl.rollback import choose_prefix, offline_diagnosis

CUTS = ["strict", "k0", "k1", "k2", "k4", "k8", "cycle", "oracle_reference"]
BACKENDS = {
    "visit": {"mode": "visit"},
    "top3_soft_erase": {"mode": "beam_soft", "score_cap": 108, "erase": True},
}


def make_cases():
    root = Path("outputs/phase2_feedback/run_001")
    cases = {}
    for name in ["dev_candidates", "dev_generated", "test_id", "test_ood", "test_generated"]:
        if name == "dev_generated":
            rows = read_jsonl(
                "outputs/seven_methods/run_001/evaluation/dev_generated_self_traces.jsonl"
            )
            cases[name] = [
                {
                    "id": r["id"],
                    "grid": r["grid"],
                    "candidate": r["candidate"],
                    "model_feedback": r["prediction"],
                }
                for r in rows
                if r["gold"] != "valid"
            ]
        else:
            filename = {
                "dev_candidates": "dev_candidates",
                "test_id": "test_id_candidates",
                "test_ood": "test_ood_candidates",
                "test_generated": "test_id_generated",
            }[name]
            rows = read_jsonl(root / (filename + "_traces.jsonl"))
            cases[name] = [
                {
                    "id": r["id"],
                    "grid": r["grid"],
                    "candidate": r["candidate"],
                    "model_feedback": r["feedback"],
                }
                for r in rows
                if r["method"] == "self" and r["gold"] != "valid"
            ]
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/phase2_reproduction/run/train/flan_t5_base/best"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new output directory")
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    expected = json.loads(Path("part3/best_overall.json").read_text())[
        "generator_checkpoint_sha256"
    ]
    if sha(args.checkpoint / "model.safetensors") != expected:
        raise ValueError("Saved feedback checkpoint differs")
    cases = make_cases()
    args.out.mkdir(parents=True)
    write_json(
        args.out / "protocol.json",
        {
            "cuts": CUTS,
            "backends": BACKENDS,
            "feedback": ["model", "executor"],
            "counts": {s: len(rows) for s, rows in cases.items()},
            "weights_sha256": expected,
            "device": "cpu",
            "max_new_actions": 36,
            "selection": "for each source/backend: dev mean unconditional optimality, mean success, fewer score rows, earlier cut; oracle excluded",
            "prefix_checks": "k groups explicitly execute the retained prefix and shorten it if illegal; strict leaves old behavior unchanged",
            "oracle_reference": "offline labeled reference, not eligible for deployment or method selection",
            "code_sha256": {
                f: sha(f) for f in [__file__, "ppnl/rollback.py", "ppnl/repair_variants.py"]
            },
            "cases_sha256": {
                s: hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
                for s, rows in cases.items()
            },
            "status": "reused exploratory cases, no training, no test-based retuning",
        },
    )
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
    selections = {}
    start = time.perf_counter()
    for split, rows in cases.items():
        grids = [Grid.from_dict(r["grid"]) for r in rows]
        truth = [oracle.check(g, r["candidate"]).label() for g, r in zip(grids, rows)]
        labels = [
            {
                "model": offline_diagnosis(g, r["candidate"], r["model_feedback"]),
                "executor": offline_diagnosis(g, r["candidate"], t),
            }
            for g, r, t in zip(grids, rows, truth)
        ]
        write_jsonl(
            args.out / f"{split}_offline_analysis.jsonl",
            [{"id": r["id"], **l} for r, l in zip(rows, labels)],
        )
        reports[split] = {}
        for source in ["model", "executor"]:
            reports[split][source] = {}
            for backend, spec in BACKENDS.items():
                reports[split][source][backend] = {}
                for cut in CUTS:
                    outputs = []
                    traces = []
                    counts = Counter()
                    before = dict(scorer.usage)
                    for r, g, t, label in zip(rows, grids, truth, labels):
                        fb = r["model_feedback"] if source == "model" else t
                        usage = {"executor_calls": 0}
                        parsed = parse_verdict(fb)
                        if cut == "strict":
                            revised, status, telemetry = repair(g, r["candidate"], fb, scorer, spec)
                            prepared, _ = repaired_prefix(g, r["candidate"], fb)
                            usage["keep_actions"] = len(prepared[0]) if prepared else None
                        else:
                            if cut == "oracle_reference" and parsed is not None and not parsed[0]:
                                keep = min(
                                    parsed[1] - 1,
                                    len(r["candidate"].split()),
                                    label[source]["latest_optimal_legal_prefix"],
                                )
                                prefix = " ".join(r["candidate"].split()[:keep])
                                usage = {
                                    "status": "ready",
                                    "keep_actions": keep,
                                    "executor_calls": 0,
                                    "oracle": True,
                                }
                            else:
                                prefix, usage = choose_prefix(
                                    g,
                                    r["candidate"],
                                    fb,
                                    int(cut[1:]) if cut.startswith("k") else 0,
                                    cut == "cycle",
                                )
                            if prefix is None:
                                revised = r["candidate"]
                                status = usage["status"]
                                telemetry = {"score_rows": 0}
                            else:
                                # Controller start position, not a new claim about V's output.
                                control = f"invalid | step={len(prefix.split())+1} | reason=goal_not_reached"
                                revised, status, telemetry = repair(
                                    g, prefix, control, scorer, spec
                                )
                        oracle.check(g, revised)
                        outputs.append(revised)
                        counts["score_rows"] += telemetry["score_rows"]
                        counts["prefix_executor_calls"] += usage["executor_calls"]
                        counts[status] += 1
                        traces.append(
                            {
                                **r,
                                "source_feedback": fb,
                                "gold_feedback": t,
                                "plan": revised,
                                "status": status,
                                "rollback": usage,
                                "score_rows": telemetry["score_rows"],
                            }
                        )
                    metrics = plan_metrics(grids, outputs)
                    reports[split][source][backend][cut] = {
                        "metrics": metrics,
                        "counts": dict(counts),
                        "policy_usage": {k: scorer.usage[k] - before[k] for k in before},
                    }
                    write_jsonl(args.out / f"{split}_{source}_{backend}_{cut}.jsonl", traces)
                    write_json(args.out / "results.json", reports)
                print(
                    split,
                    source,
                    backend,
                    {
                        k: (v["metrics"]["success_rate"], v["metrics"]["optimality"])
                        for k, v in reports[split][source][backend].items()
                    },
                    flush=True,
                )
        if split == "dev_generated":
            for source in ["model", "executor"]:
                selections[source] = {}
                for backend in BACKENDS:

                    def rank(cut):
                        a = reports["dev_candidates"][source][backend][cut]
                        b = reports["dev_generated"][source][backend][cut]
                        return (
                            (a["metrics"]["optimality"] + b["metrics"]["optimality"]) / 2,
                            (a["metrics"]["success_rate"] + b["metrics"]["success_rate"]) / 2,
                            -a["counts"]["score_rows"] - b["counts"]["score_rows"],
                        )

                    best = max(CUTS[:-1], key=rank)
                    selections[source][backend] = {"cut": best, "rank": rank(best)}
            write_json(args.out / "selection.json", selections)
            print("SELECTED", selections, flush=True)
    write_json(
        args.out / "completed.json",
        {
            "completed": True,
            "device": "cpu",
            "gpu_calls": 0,
            "elapsed_seconds": time.perf_counter() - start,
            "selection": selections,
        },
    )


if __name__ == "__main__":
    main()
