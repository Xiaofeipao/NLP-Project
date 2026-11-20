"""Evaluate the two selected checkpoints on all directions from each frozen dev G state."""
import argparse
from collections import Counter, defaultdict
import gc
from pathlib import Path

import torch
from ppnl.data import read_jsonl, write_json, write_jsonl
from ppnl.experiment_freeze import file_hash
from ppnl.policy import NAMES
from state_baselines.run import CONFIG
from step_data_v3 import g_probes
from step_data_v3.five_class import FiveClassRunner
from step_data_v3.model_io import check_checkpoint
from step_data_v3.schema import State
from step_data_v3.train_gpu import TextVerifierRunner, local_v, split


def evaluate(runner, rows):
    check_checkpoint(runner, g_probes.template_hash())
    _, predictions = local_v(runner, rows, lambda: None)
    grouped = defaultdict(list)
    for row, record in zip(rows, predictions, strict=True):
        grouped[row["state_id"]].append((row, record["prediction"]))
    if len(grouped) != 100:
        raise ValueError("Expected all 100 development G states")
    total_correct = all_correct = same_class = arrival_groups = arrival_only = 0
    valid_fp = Counter()
    recall = Counter()
    gold = Counter()
    for pairs in grouped.values():
        if sorted(row["candidate"] for row, _ in pairs) != sorted(NAMES):
            raise ValueError("Expected four distinct directions per state")
        hits = sum(row["target"] == pred for row, pred in pairs)
        total_correct += hits
        all_correct += hits == 4
        same_class += len({pred for _, pred in pairs}) == 1
        if any(row["target"] == "valid" for row, _ in pairs):
            arrival_groups += 1
            arrival_only += all((row["target"] == "valid") == (pred == "valid") for row, pred in pairs)
        for row, pred in pairs:
            target = row["target"]
            gold[target] += 1
            recall[target] += pred == target
            if pred == "valid" and target != "valid":
                valid_fp[target] += 1
    return {
        "correct_steps": [total_correct, len(rows)], "states_all_four_correct": [all_correct, len(grouped)],
        "states_four_identical_predictions": [same_class, len(grouped)],
        "adjacent_states_only_arriving_direction_marked_valid": [arrival_only, arrival_groups],
        "false_valid_by_true_class": dict(valid_fp),
        "recall_by_class": {c: [recall[c], n] for c, n in gold.items()},
    }, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "old", "new", "out"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    path = args.data / "diagnostics" / "g_dev_four.jsonl"
    rows = read_jsonl(path)
    settings = dict(CONFIG["training"], precision="bfloat16" if torch.cuda.is_bf16_supported() else "float32",
                    eval_batch_size=16, max_input_tokens=128, max_target_tokens=8)
    results = {"probe_sha256": file_hash(path), "shared_with_formal_dev_V": sum(
        row["in_same_split_supervised_V"] for row in rows)}
    for name, cls, checkpoint in (("text_epoch_2", TextVerifierRunner, args.old),
                                  ("five_class_selected", FiveClassRunner, args.new)):
        runner = cls(str(checkpoint), dict(CONFIG, training=settings))
        results[name], records = evaluate(runner, rows)
        write_jsonl(args.out / f"four_moves_{name}.jsonl", records)
        if cls is FiveClassRunner:
            dev = [r for r in split(args.data,"dev") if r["mode"]=="v_step"]
            states = {r["id"]:State.from_record(r) for r in read_jsonl(args.data/"states.jsonl")}
            saved = read_jsonl(args.out/f"train_dev_epoch_{checkpoint.name.rsplit('-',1)[1]}.jsonl")
            decisions = g_probes.verify_moves(runner,[(states[r["state_id"]],r["candidate"]) for r in dev])
            assert [(r["id"],p) for r,p in zip(dev,decisions,strict=True)]==[(r["id"],r["prediction"]) for r in saved]
            features = g_probes.prepare_training(runner,args.data)
            indices = [next(i for i,f in enumerate(features) if sum(f["policy_actions"])>1)]
            indices.extend(next(i for i,f in enumerate(features) if f["labels"]==k) for k in range(5))
            batch = runner.collate([features[i] for i in indices])
            with torch.inference_mode(),runner.autocast():
                public_loss = g_probes.batch_loss(runner,batch)
                direct_loss = runner.batch_loss(batch)
            torch.testing.assert_close(public_loss,direct_loss)
            results["public_api_check"] = {"selected_dev_decisions_reproduced":len(saved),
                "five_class_features":len(features),"mixed_public_loss":public_loss.item(),
                "mixed_direct_loss":direct_loss.item(),"verifier_loss":runner.model.config.ppnl_verifier_loss}
        del runner
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.out / "four_moves_comparison.json", results)


if __name__ == "__main__":
    main()
