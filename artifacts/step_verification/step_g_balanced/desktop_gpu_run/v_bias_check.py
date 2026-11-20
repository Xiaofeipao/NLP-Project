"""Read-only diagnostics for the frozen balanced G/V checkpoint, on train and dev."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch
from transformers.modeling_outputs import BaseModelOutput

from ppnl.data import read_jsonl
from ppnl.modeling import Seq2Seq
from state_baselines.run import CONFIG
from step_data_v3 import g_probes
from step_data_v3.model_io import check_checkpoint, read_split
from step_data_v3.shortcut_audit import metrics
from step_data_v3.single import LABELS


def summarize(rows, predictions):
    gold = [r["target"] for r in rows]
    confusion = defaultdict(Counter)
    for label, prediction in zip(gold, predictions, strict=True):
        confusion[label][prediction] += 1
    return dict(metrics(gold, predictions), predictions=dict(Counter(predictions)), confusion=dict(confusion))


def controls(rows, predictions, groups):
    by_id = {r["id"]: (r, p) for r, p in zip(rows, predictions, strict=True)}
    result = defaultdict(Counter)
    for g in groups:
        kind = g["kind"]
        if kind not in {"swap_walls", "change_wall", "change_goal", "change_edge"}:
            continue
        by_move = defaultdict(list)
        for example in g["rows"]:
            row, pred = by_id[example]
            by_move[row["candidate"]].append((row["target"], pred))
        for pair in by_move.values():
            (t1, p1), (t2, p2) = pair
            assert t1 != t2
            cell = result[kind]
            cell["pairs"] += 1
            cell["different_prediction"] += p1 != p2
            cell["both_correct"] += t1 == p1 and t2 == p2
            cell["both_valid_prediction"] += p1 == p2 == "valid"
    return dict(result)


@torch.inference_mode()
def label_scores(runner, rows):
    tokenizer, model = runner.tokenizer, runner.model
    targets = tokenizer(text_target=list(LABELS), padding=True, return_tensors="pt")
    target = targets.input_ids.to(runner.device)
    mask = targets.attention_mask.to(runner.device).bool()
    lengths = mask.sum(-1)
    first_ids = target[:, 0]
    assert len(set(first_ids.tolist())) == len(LABELS)
    stats = defaultdict(Counter)
    methods = defaultdict(list)
    probabilities = defaultdict(list)
    for offset in range(0, len(rows), 8):
        part = rows[offset:offset+8]
        n = len(part)
        inputs = tokenizer([r["input"] for r in part], padding=True, pad_to_multiple_of=8,
                           return_tensors="pt").to(runner.device)
        reps = len(LABELS)
        with runner.autocast():
            encoded = model.get_encoder()(**inputs).last_hidden_state
            outputs = model(encoder_outputs=BaseModelOutput(last_hidden_state=encoded.repeat_interleave(reps, 0)),
                            attention_mask=inputs.attention_mask.repeat_interleave(reps, 0),
                            labels=target.masked_fill(~mask, -100).repeat(n, 1), use_cache=False)
        logits = outputs.logits.float().reshape(n, reps, target.shape[1], -1)
        first = logits[:, 0, 0, :][:, first_ids]
        logp = logits.log_softmax(-1).gather(-1, target.expand(n, -1, -1).unsqueeze(-1)).squeeze(-1)
        sequence = logp.masked_fill(~mask, 0).sum(-1)
        scores = {"first_token": first, "sequence_sum": sequence, "sequence_mean": sequence / lengths,
                  "first_token_times_label_length": first + lengths.float().log()}
        for method, values in scores.items():
            methods[method].extend(LABELS[i] for i in values.argmax(-1).tolist())
        teacher = logits.argmax(-1)
        for i, row in enumerate(part):
            label = row["target"]
            j = LABELS.index(label)
            k = lengths[j].item()
            totals = stats[label]
            totals["rows"] += 1
            totals["first_token_correct_full_vocabulary"] += int(teacher[i, j, 0] == target[j, 0])
            totals["suffix_tokens"] += k - 1
            totals["suffix_correct_given_gold_prefix"] += (teacher[i, j, 1:k] == target[j, 1:k]).sum().item()
            probabilities[label].append(first[i].softmax(-1).tolist())
    return {"lengths_including_eos": dict(zip(LABELS, lengths.tolist(), strict=True)),
            "conditional_first_token_probability_among_five_by_gold": {
                label: dict(zip(LABELS, torch.tensor(values).mean(0).tolist(), strict=True))
                for label, values in probabilities.items()},
            "teacher_forcing": dict(stats),
            "decision_diagnostics": {m: summarize(rows, preds) for m, preds in methods.items()}}


def run(args):
    settings = dict(CONFIG["training"], precision="bfloat16", eval_batch_size=16,
                    max_input_tokens=128, max_target_tokens=8)
    runner = Seq2Seq(str(args.checkpoint), dict(CONFIG, training=settings))
    check_checkpoint(runner, g_probes.template_hash())
    runner.model.eval()
    groups = json.loads((args.data / "control_groups.json").read_bytes())
    stored = {r["id"]: r["prediction"] for r in read_jsonl(args.recorded_dev)}
    result = {"checkpoint": args.checkpoint.name}
    for split in ("dev", "train"):
        all_rows = read_split(args.data, split, version=g_probes.VERSION, fingerprint=g_probes.template_hash())
        rows = [r for r in all_rows if r["mode"] == "v_step"]
        preds = [p.strip() for p in runner.predict([r["input"] for r in rows])]
        result[split] = {"greedy": summarize(rows, preds), "paired_controls": controls(rows, preds, groups[split])}
        if split == "dev":
            result[split]["reproduced_archived_predictions"] = sum(p == stored[r["id"]] for r, p in zip(rows, preds))
            result[split]["scores"] = label_scores(runner, rows)
        print(split, result[split]["greedy"]["correct"], result[split]["greedy"]["predictions"], flush=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "checkpoint", "recorded-dev", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    run(parser.parse_args())
