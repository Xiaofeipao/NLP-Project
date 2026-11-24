"""Run a fixed coordinate checkpoint on blind paths with and without the endpoint check."""
import argparse
import json
from pathlib import Path

import torch

from ppnl.core import Grid
from ppnl.data import read_jsonl
from ppnl.experiment_freeze import file_hash
from . import g_probes, last_move
from .coordinate_paths import destination
from .model_io import check_checkpoint
from .path_validation import load_runner
from .single import LABELS


def original_prompt(state, action):
    return last_move.step_input(state, action, explicit_vocabulary=True)


def lookup_both(runner, unique, renderer):
    keys = sorted(unique)
    valid_index = LABELS.index("valid")
    labels, maximum = {}, 0
    size = runner.config["training"]["eval_batch_size"]
    for pos in range(0, len(keys), size):
        group = keys[pos:pos + size]
        observations = [unique[k] for k in group]
        texts = [renderer(s, a) for s, a in observations]
        batch = runner.tokenizer(texts, padding=True, truncation=False, return_tensors="pt").to(runner.device)
        n = batch.input_ids.shape[1]
        if n > runner.input_limit:
            raise ValueError(f"Coordinate prompt requires {n} tokens")
        maximum = max(maximum, n)
        with torch.inference_mode(), runner.autocast():
            scores = runner.first_logits(batch)[:, runner.verdict_ids]
            both = scores.float().cpu()
        arrivals = [destination(s.rows, s.cols, s.obstacles, s.current, a) == s.goal for s, a in observations]
        raw = both.argmax(dim=1).tolist()
        checked_scores = both.clone()
        checked_scores[:, valid_index].masked_fill_(~torch.tensor(arrivals, dtype=torch.bool), -torch.inf)
        checked = checked_scores.argmax(dim=1).tolist()
        for k, before, after, allowed in zip(group, raw, checked, arrivals, strict=True):
            if before != after and (before != valid_index or allowed):
                raise ValueError("Endpoint check changed an eligible prediction")
            labels[k] = (LABELS[before], LABELS[after])
    return labels, maximum


def evaluate(args):
    torch.set_num_threads(2)
    runner = load_runner(args.checkpoint, False, "bfloat16", first_token=True)
    runner.model.eval()
    renderer = original_prompt if args.original else g_probes.step_input
    fingerprint = last_move.template_hash(renderer)
    check_checkpoint(runner, fingerprint)
    unique, parts = {}, {}
    for part in ("single", "double"):
        cases = read_jsonl(args.blind / (part + ".jsonl"))
        keys_by_case = []
        for item in cases:
            if set(item) != {"id", "grid", "candidate", "current_before"}:
                raise ValueError("Unexpected evaluation field")
            grid, plan = Grid.from_dict(item["grid"]), item["candidate"]
            keys = []
            for current, (state, a) in zip(item["current_before"], last_move.path_steps(grid, plan), strict=True):
                if tuple(current) != state.current:
                    raise ValueError(f"Pre-move coordinates changed for {item['id']}")
                key = state.id, a
                unique[key] = state, a
                keys.append(key)
            keys_by_case.append(keys)
        if len({r["id"] for r in cases}) != len(cases) or len(cases) != 200:
            raise ValueError(f"Unexpected {part} IDs")
        parts[part] = cases, keys_by_case
    prefix = args.prefix.read_text(encoding="utf-8").rstrip() if args.prefix else ""
    def prompt(state, action):
        query = renderer(state, action)
        return prefix + "\n\n" + query if prefix else query
    if prefix:
        runner.input_limit = 512
    cache, maximum = lookup_both(runner, unique, prompt)
    args.out.mkdir(parents=True, exist_ok=True)
    for part, (cases, keys) in parts.items():
        with (args.out / (part + ".jsonl")).open("w", encoding="utf-8", newline="\n") as out:
            for case, seq in zip(cases, keys, strict=True):
                record = dict(id=case["id"], step_labels=[cache[k] for k in seq])
                out.write(json.dumps(record, separators=(",", ":")) + "\n")
    receipt = dict(template_sha256=fingerprint, model_sha256=file_hash(args.checkpoint / "model.safetensors"),
                   max_input_tokens=maximum, unique_step_inputs=len(cache), source_prompt=args.original,
                   single_forward_pass_per_unique_input=True, official_executor_calls=0,
                   unit="one candidate move", case_verdicts=False,
                   prompt_prefix_sha256=file_hash(args.prefix) if args.prefix else None)
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("checkpoint", "blind", "out"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--original", action="store_true")
    parser.add_argument("--prefix", type=Path)
    evaluate(parser.parse_args())
