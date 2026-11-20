"""Offline tokenizer, tied-action loss, real collation and one CPU backward pass."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from ppnl.core import Grid
from ppnl.data import read_jsonl
from ppnl.modeling import Seq2Seq, action_set_loss
from ppnl.policy import NAMES, state_prompt
from ppnl.prompts import prompt
from state_baselines.step_verification import step_prompt
from .audit import require
from .build import write_object
from .model_io import StatePolicy, batch_loss, prepare_training, read_split, verify_moves, verify_plans
from .schema import State, render, step_labels, template_hash


def tokenizer_check(tokenizer, rows):
    lengths, targets = defaultdict(list), defaultdict(list)
    encoded = tokenizer([r["input"] for r in rows], truncation=False)["input_ids"]
    labels = tokenizer(text_target=[r["target"] for r in rows], truncation=False)["input_ids"]
    decoded = tokenizer.batch_decode(labels, skip_special_tokens=True)
    for row, inp, ids, out in zip(rows, encoded, labels, decoded, strict=True):
        require(row["target"] == out and ids[-1] == tokenizer.eos_token_id, row["id"])
        require(len(inp) <= 512 and len(ids) <= 128 and tokenizer.unk_token_id not in inp, row["id"])
        lengths[row["mode"]].append(len(inp))
        targets[row["mode"]].append(len(ids))
    return {k: {"rows": len(v), "max_input": max(v), "mean_input": round(sum(v)/len(v), 2),
                "max_target": max(targets[k])} for k,v in lengths.items()}


def objective_check(torch):
    # A decoder assigning the same total probability to down/right must get the same loss.
    ids = [0, 1, 2, 3]
    allowed = torch.tensor([[False, True, False, True]])
    labels = torch.tensor([[1]])
    def loss(values):
        return action_set_loss(torch.tensor([[values]]), labels, ids, allowed, "optimal_actions",
                               split_verification_loss=True).item()
    a, b = loss([0., 4., 0., -1.]), loss([0., -1., 0., 4.])
    wrong = loss([4., -1., 0., 0.])
    require(abs(a-b) < 1e-6 and wrong > a, "tied-action probability mass")
    return True


def previous_prompt_lengths(tokenizer, rows, registry):
    by_mode = defaultdict(list)
    for row in rows:
        state = registry[row["state_id"]]
        grid, _ = state.witness()
        mode = row["mode"]
        if mode == "g_next":
            old = state_prompt(grid, state.current, local_status=True)
        elif mode == "v_step":
            old = step_prompt(grid, state.current, row["candidate"] or None)
        else:
            old = prompt("verification", grid, row["candidate"])
        by_mode[mode].append(old)
    return {m: round(sum(map(len, tokenizer(texts, truncation=False)["input_ids"]))/len(texts), 2)
            for m, texts in by_mode.items()}


def cpu_runner(tokenizer_file):
    import torch
    from transformers import (DataCollatorForSeq2Seq, PreTrainedTokenizerFast, T5Config,
                              T5ForConditionalGeneration)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file), eos_token="</s>",
                                        pad_token="<pad>", unk_token="<unk>", model_max_length=512)
    runner = Seq2Seq.__new__(Seq2Seq)
    runner.torch, runner.tokenizer, runner.device, runner.precision = torch, tokenizer, "cpu", "float32"
    runner.config = {"training": {"eval_batch_size": 4}}
    runner.input_limit, runner.output_limit, runner.policy_loss = 512, 128, "optimal_actions"
    action_ids = [tokenizer.encode(a, add_special_tokens=False) for a in NAMES]
    require(all(len(a) == 1 for a in action_ids), "four one-token directions")
    runner.action_ids = [a[0] for a in action_ids]
    model_config = T5Config(vocab_size=len(tokenizer), d_model=16, d_kv=8, d_ff=32, num_layers=1,
                            num_decoder_layers=1, num_heads=2, decoder_start_token_id=tokenizer.pad_token_id,
                            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    runner.model = T5ForConditionalGeneration(model_config)
    runner._collator = DataCollatorForSeq2Seq(tokenizer, model=runner.model)
    return runner


def run(dataset, tokenizer_file):
    runner = cpu_runner(tokenizer_file)
    torch, tokenizer = runner.torch, runner.tokenizer
    stats = {s: tokenizer_check(tokenizer, read_split(dataset,s)) for s in ("train","dev","test_id","test_ood")}
    try:
        StatePolicy(runner)
    except ValueError:
        pass
    else:
        raise ValueError("Unversioned checkpoint accepted")
    features = prepare_training(runner, dataset)
    rows = read_split(dataset, "train")
    masks = Counter(sum(x["policy_actions"]) for x in features)
    selects = [next(i for i,r in enumerate(rows) if len(r.get("optimal_actions", ())) > 1),
               next(i for i,r in enumerate(rows) if r["mode"] == "v_step" and r["target"] != "valid"),
               next(i for i,r in enumerate(rows) if r["mode"] == "v_plan" and r["target"] == "valid")]
    batch = runner.collate([features[i] for i in selects])
    loss = batch_loss(runner, batch)
    loss.backward()
    gradients = [p.grad for p in runner.model.parameters() if p.grad is not None]
    require(torch.isfinite(loss).item() and all(torch.isfinite(g).all() for g in gradients)
            and any(torch.count_nonzero(g) for g in gradients), "finite nonzero gradients")

    # Capture inputs sent by each deployment API and check exact parity with stored training text.
    registry = {r["id"]: State.from_record(r) for r in read_jsonl(dataset / "states.jsonl")}
    g, vs, vp = (rows[i] for i in selects)
    sg, ss, sp = (registry[r["state_id"]] for r in (g, vs, vp))
    seen = []
    class CaptureTokenizer:
        def __call__(self, texts, **kwargs):
            seen.extend(texts)
            return tokenizer(texts, **kwargs)
    runner.tokenizer = CaptureTokenizer()
    scores = StatePolicy(runner).scores([(sg.grid(), sg.current)])
    require(seen == [g["input"]] and len(scores) == 1 and len(scores[0]) == 4, "policy inference input")
    runner.tokenizer = tokenizer
    # V adapters hand text to the existing runner. Also execute one real greedy decode below.
    def capture(texts, check):
        check()
        return texts
    predict = runner.predict
    runner.predict = capture
    grid, _ = ss.witness()
    require(verify_moves(runner, [(grid, ss.current, vs["candidate"])]) == [vs["input"]], "local inference input")
    require(verify_plans(runner, [sp.grid()], [vp["candidate"]]) == [vp["input"]], "plan inference input")
    runner.predict = predict
    runner.output_limit = max(len(tokenizer.encode(r["target"])) for r in (vs, vp))
    require(len(verify_plans(runner, [sp.grid()], [vp["candidate"]])) == 1, "CPU teacher-free decode")
    test_grid = Grid(3, 3, ((1,0),), (0,0), (0,2))
    away, goal = State.from_grid(test_grid), State.from_grid(test_grid, test_grid.goal)
    require(step_labels(away,"")[0].startswith("invalid | step=1") and step_labels(goal,"")[0] == "valid"
            and step_labels(away,"right")[0].startswith("invalid | step=2")
            and render("v_step", away, "") != render("v_step", away, "stop"), "stop semantics")
    return {"model": "random 1-layer T5, CPU integration check", "template_sha256": template_hash(),
            "tokenizer_sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
            "tokens": stats, "older_renderer_mean_input_on_identical_train_examples": previous_prompt_lengths(tokenizer, rows, registry),
            "all_target_text_roundtrips": True, "input_unknown_tokens": 0,
            "G_mask_cardinalities_with_zero_for_V": dict(masks), "tied_direction_loss_checked": objective_check(torch),
            "finite_backward_mixed_batch": True, "train_inference_prompt_parity": ["g_next", "v_step", "v_plan"],
            "cpu_greedy_decode": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True, help="Path to the Flan-T5-base tokenizer.json")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = run(args.data, args.tokenizer)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_object(args.out, report)
    print(json.dumps(report, indent=2))
