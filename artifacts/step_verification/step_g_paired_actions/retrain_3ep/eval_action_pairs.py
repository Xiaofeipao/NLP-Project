"""Compare four-direction probes on the old and new selected V checkpoints."""
import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path

import torch
from ppnl.data import read_jsonl, write_json, write_jsonl
from ppnl.experiment_freeze import file_hash
from ppnl.policy import NAMES
from state_baselines.run import CONFIG
from step_data_v3 import g_probes
from step_data_v3.five_class import FiveClassRunner
from step_data_v3.schema import State
from step_data_v3.train_gpu import TextVerifierRunner


def summarize(rows, predictions):
    by_state = defaultdict(list)
    confusions = defaultdict(Counter)
    for r, p in zip(rows, predictions, strict=True):
        by_state[r["state_id"]].append((r["candidate"], r["target"], p))
        confusions[r["target"]][p] += 1
    assert all({a for a, _, _ in cases} == set(NAMES) and len(cases) == 4 for cases in by_state.values())
    arriving = [v for v in by_state.values() if any(t == "valid" for _, t, _ in v)]
    return {
        "steps_correct": [sum(c[t] for t, c in confusions.items()), len(rows)],
        "states_all_four_correct": [sum(all(y == p for _, y, p in v) for v in by_state.values()), len(by_state)],
        "states_same_prediction_for_all_moves": [sum(len({p for _, _, p in v}) == 1 for v in by_state.values()), len(by_state)],
        "states_only_arriving_move_called_valid": [
            sum(all((y == "valid") == (p == "valid") for _, y, p in v) for v in arriving), len(arriving)
        ],
        "false_valid": sum(c["valid"] for t, c in confusions.items() if t != "valid"),
        "confusion": confusions,
    }


def selected(directory):
    choice = json.loads((directory / "selection.json").read_bytes())["selected"]
    return directory / choice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "new", "old-text", "old-head", "out"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    paths = {k: args.data / "diagnostics" / f"{k}_dev_four.jsonl" for k in ("g", "v")}
    probes = {k: read_jsonl(p) for k, p in paths.items()}
    union = {(r["state_id"], r["candidate"]): r for vs in probes.values() for r in vs}
    states = {r["id"]: State.from_record(r) for r in read_jsonl(args.data / "states.jsonl")}
    observations = [(states[s], a) for s, a in union]
    assert [g_probes.step_input(s, a) for s, a in observations] == [r["input"] for r in union.values()]
    probes["v_only"] = [r for r in probes["v"] if r["source_g_id"] is None]
    args.out.mkdir(parents=True, exist_ok=False)
    settings = dict(CONFIG["training"], precision="bfloat16" if torch.cuda.is_bf16_supported() else "float32",
                    eval_batch_size=16, max_input_tokens=128, max_target_tokens=8)
    report = {"probe_hashes": {k: file_hash(p) for k, p in paths.items()},
              "shared_with_formal_dev_V": {k: sum(r["in_same_split_supervised_V"] for r in rs) for k, rs in probes.items()}}
    for k, cls, checkpoint in (
        ("old_text", TextVerifierRunner, args.old_text), ("new_text", TextVerifierRunner, selected(args.new / "text")),
        ("new_text_epoch_2", TextVerifierRunner, args.new / "text" / "checkpoint-epoch-2"),
        ("old_head", FiveClassRunner, args.old_head), ("new_head", FiveClassRunner, selected(args.new / "five_class")),
    ):
        runner = cls(str(checkpoint), dict(CONFIG, training=settings, local_files_only=True))
        decisions = g_probes.verify_moves(runner, observations)
        predictions = dict(zip(union, decisions, strict=True))
        report[k] = {"checkpoint": str(checkpoint)}
        for name, rows in probes.items():
            predicted = [predictions[r["state_id"], r["candidate"]] for r in rows]
            report[k][name] = summarize(rows, predicted)
            if name != "v_only":
                write_jsonl(args.out / f"{k}_{name}.jsonl", [
                    dict(id=r["id"], state_id=r["state_id"], candidate=r["candidate"], target=r["target"], prediction=p)
                    for r, p in zip(rows, predicted, strict=True)
                ])
        del runner
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.out / "action_comparison.json", report)


if __name__ == "__main__":
    main()
