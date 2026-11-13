"""Train the three state-policy baselines and freeze development-only selection."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time

from ppnl.core import Grid, OfficialOracle
from ppnl.data import read_jsonl, write_json, write_jsonl
from ppnl.evaluation import plan_metrics
from ppnl.experiment_freeze import audit_splits, file_hash
from ppnl.modeling import Seq2Seq, gpu_budget, require_cuda
from ppnl.optimization import accumulated_batches
from ppnl.policy import NAMES, NeuralPolicy, greedy_plans
from state_baselines.workflow import compare, verifier_report, verify


PROTOCOL = Path(__file__).with_name("protocol.json")
CONFIG = json.loads(PROTOCOL.read_bytes())
SOURCES = (
    PROTOCOL, Path(__file__), Path(__file__).with_name("workflow.py"),
    *(Path("ppnl") / f"{name}.py" for name in (
        "core", "data", "evaluation", "experiment_freeze", "local_finish",
        "modeling", "optimization", "policy", "prompts")),
)


def grids_for(rows):
    return [Grid.from_dict(row["grid"]) for row in rows]


def unique_grids(rows):
    return list(dict.fromkeys(grids_for(rows)))


def identity(args):
    return {
        "source": {path.name: file_hash(path) for path in SOURCES},
        "data": {name: file_hash(args.data / f"{name}.jsonl")
                 for name in ("train", "dev", "test_id", "test_ood")},
        "manifest": file_hash(args.data / "manifest.json"),
        "executor": OfficialOracle(args.executor).sha256,
    }


def audit(args):
    result = audit_splits(args.data)
    oracle = OfficialOracle(args.executor)
    manifest = json.loads((args.data / "manifest.json").read_bytes())
    if oracle.sha256 != manifest["executor_sha256"]:
        raise ValueError("Executor does not match the dataset manifest")
    for split, report in result["splits"].items():
        rows = read_jsonl(args.data / f"{split}.jsonl")
        counts = Counter()
        for grid, row in zip(grids_for(rows), rows, strict=True):
            if "candidate" in row:
                verdict = oracle.check(grid, row["candidate"])
                if verdict.label() != row["label"]:
                    raise ValueError(f"{split}: stale label for {row['id']}")
                if split.startswith("test_") and not verdict.valid:
                    expected = 1 if split == "test_id" else 2
                    if row["edit_count"] != expected:
                        raise ValueError(f"{split}: wrong edit count for {row['id']}")
                counts[verdict.reason or "valid"] += 1
        report["labels"] = dict(counts)
        report["environments"] = len(unique_grids(rows))
    result["identity"] = identity(args)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "dataset_audit.json", result)
    print({name: {"environments": value["environments"], "labels": value["labels"]}
           for name, value in result["splits"].items()}, flush=True)


def dev_eval(runner, rows, oracle, check, pilot):
    policy_rows = [row for row in rows if row["task"] == "generation"]
    verification = [row for row in rows if row["task"] == "verification"]
    if pilot:
        policy_rows, verification = policy_rows[:4], verification[:8]
    scorer = NeuralPolicy(runner, check)
    observations = [(Grid.from_dict(row["grid"]), row["policy_position"]) for row in policy_rows]
    scores = scorer.scores(observations)
    action_accuracy = sum(
        NAMES[max(range(4), key=score.__getitem__)] in row["optimal_actions"]
        for score, row in zip(scores, policy_rows, strict=True)
    ) / len(policy_rows)
    grids = unique_grids(rows) if not pilot else unique_grids(policy_rows)
    plans = greedy_plans(grids, scorer, masked=False, max_steps=CONFIG["max_steps"])
    for grid, plan in zip(grids, plans, strict=True):
        oracle.check(grid, plan)
    predictions = runner.predict([row["input"] for row in verification], check)
    return {
        "planning": plan_metrics(grids, plans),
        "action_accuracy": action_accuracy,
        "verification": verifier_report(verification, predictions),
    }, {
        "planning": [{"grid": asdict(grid), "plan": plan} for grid, plan in zip(grids, plans, strict=True)],
        "verification": [{"id": row["id"], "target": row["target"], "prediction": prediction}
                         for row, prediction in zip(verification, predictions, strict=True)],
    }


def rank(metrics):
    return (metrics["planning"]["success_rate"], metrics["verification"]["macro_f1"],
            metrics["planning"]["optimality"])


def train(args):
    pilot = args.command == "pilot"
    torch = require_cuda()
    torch.set_num_threads(1)
    settings = CONFIG["training"]
    model = CONFIG["models"][args.model]
    rows = read_jsonl(args.data / "train.jsonl")
    dev = read_jsonl(args.data / "dev.jsonl")
    expected = json.loads((args.out / "dataset_audit.json").read_bytes())["identity"]
    if identity(args) != expected:
        raise ValueError("Data or code changed since audit")
    out = args.out / args.command / args.model
    out.mkdir(parents=True)
    started = time.perf_counter()
    oracle = OfficialOracle(args.executor)
    revision_file = args.pretrained / "upstream_revision.txt"
    revision = revision_file.read_text().strip() if revision_file.exists() else args.pretrained.name
    if revision != model["revision"]:
        raise ValueError("Pretrained snapshot revision does not match the protocol")
    pretrained = {p.name: file_hash(p) for p in args.pretrained.iterdir()
                  if p.is_file() and not p.name.startswith(".")}
    with gpu_budget(args.ledger, 20, f"phase1:{args.command}:{args.model}") as check:
        runner = Seq2Seq(str(args.pretrained), CONFIG)
        if args.model == "flan_t5_base":
            runner.model.gradient_checkpointing_enable()
            runner.model.config.use_cache = False
        parameters = list(runner.model.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"],
                                      weight_decay=settings["weight_decay"], fused=True)
        if pilot:
            rows = [r for task in ("generation", "verification")
                    for r in [r for r in rows if r["task"] == task][:16]]
        features = runner.encode_rows(rows)
        action_index = next(i for i, row in enumerate(rows) if row["task"] == "generation")
        action_batch = runner.collate([features[action_index]]).to(runner.device)
        if action_batch["labels"][0, 0].item() not in runner.action_ids:
            raise ValueError("Action target is misaligned with the first decoder position")
        if action_batch["decoder_input_ids"][0, 0].item() != runner.model.config.decoder_start_token_id:
            raise ValueError("Training and inference use different action decoder starts")
        batch_size = model["microbatch"]
        loader = torch.utils.data.DataLoader(
            features, batch_size=batch_size, shuffle=True, collate_fn=runner.collate,
            generator=torch.Generator().manual_seed(CONFIG["seed"]),
        )
        history, best, updates = [], None, 0
        probe = parameters[0].detach().flatten()[:32].clone()
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1, (1 if pilot else CONFIG["epochs"]) + 1):
            runner.model.train()
            total_loss = 0
            for _, batch, scale, update in accumulated_batches(
                    loader, CONFIG["effective_batch"] // batch_size, "examples"):
                check()
                loss = runner.loss(batch.to(runner.device))
                value = float(loss.detach())
                if not torch.isfinite(loss).item():
                    raise ValueError("Non-finite training loss")
                total_loss += value * len(batch["labels"])
                (loss * scale).backward()
                if update:
                    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    if updates == 0 and not grad_norm.item() > 0:
                        raise ValueError("First training update has no gradient")
                    optimizer.step()
                    if updates == 0 and torch.equal(probe, parameters[0].detach().flatten()[:32]):
                        raise ValueError("Optimizer step did not change the probe weights")
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    if pilot or updates % 25 == 0:
                        progress = {"epoch": epoch, "updates": updates, "loss": value,
                                    "seconds": time.perf_counter() - started}
                        write_json(out / "progress.json", progress)
                        print(args.model, progress, flush=True)
            metrics, traces = dev_eval(runner, dev, oracle, check, pilot)
            entry = {"epoch": epoch, "train_loss": total_loss / len(rows), **metrics}
            history.append(entry)
            write_json(out / f"dev_epoch_{epoch}.json", traces)
            print(args.model, entry, flush=True)
            if best is None or rank(entry) > rank(best):
                best = entry
                if not pilot:
                    runner.model.save_pretrained(out / "best")
                    runner.tokenizer.save_pretrained(out / "best")
        write_json(out / "training.json", {
            "model": args.model, "pretrained": model, "pretrained_files": pretrained, "identity": expected,
            "updates": updates, "examples": len(rows) * len(history), "history": history, "best": best,
            "gpu": torch.cuda.get_device_name(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "seconds": time.perf_counter() - started, "pilot": pilot,
            "parameters": sum(p.numel() for p in parameters),
            "trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad),
        })


def select(args):
    expected = identity(args)
    results = {name: json.loads((args.out / "train" / name / "training.json").read_bytes())
               for name in CONFIG["models"]}
    if any(r["identity"] != expected or r["updates"] != 500 for r in results.values()):
        raise ValueError("All three complete training runs must match the current protocol")
    main = max(results, key=lambda name: (*rank(results[name]["best"]), -results[name]["parameters"]))
    write_json(args.out / "selection.json", {
        "main_model": main, "development": {name: r["best"] for name, r in results.items()},
        "identity": expected,
        "weights": {name: file_hash(args.out / "train" / name / "best" / "model.safetensors")
                    for name in results},
    })


def evaluate(args):
    selection = json.loads((args.out / "selection.json").read_bytes())
    checkpoint = args.out / "train" / args.model / "best"
    if (selection["identity"] != identity(args)
            or selection["weights"][args.model] != file_hash(checkpoint / "model.safetensors")):
        raise ValueError("Evaluation differs from frozen development selection")
    if args.split == "test_ood" and args.model != selection["main_model"]:
        raise ValueError("The full OOD evaluation uses the development-selected model")
    out = args.out / args.split / args.model
    out.mkdir(parents=True)
    rows = read_jsonl(args.data / f"{args.split}.jsonl")
    oracle = OfficialOracle(args.executor)
    require_cuda().set_num_threads(1)
    with gpu_budget(args.ledger, 20, f"phase1:{args.split}:{args.model}") as check:
        runner = Seq2Seq(str(checkpoint), CONFIG)
        grids = grids_for(rows)
        metrics = {"test_status": "reused_exploratory", "model": args.model}
        if args.split == "test_id":
            plans = greedy_plans(grids, NeuralPolicy(runner, check), masked=False, max_steps=CONFIG["max_steps"])
            metrics["planning"], traces = compare(runner, grids, plans, oracle, check)
            write_jsonl(out / "planning.jsonl", [dict(id=row["id"], grid=row["grid"], **trace)
                                                 for row, trace in zip(rows, traces, strict=True)])
        else:
            metrics["planning"] = {"shared_id_environments": True}
        candidates = [row["candidate"] for row in rows]
        metrics["fixed_candidates"], traces = compare(runner, grids, candidates, oracle, check)
        write_jsonl(out / "candidates.jsonl", [dict(id=row["id"], grid=row["grid"], **trace)
                                               for row, trace in zip(rows, traces, strict=True)])
        write_json(out / "metrics.json", metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "pilot", "train", "select", "evaluate"))
    parser.add_argument("--data", type=Path, default=Path("data/policy_v3"))
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--model", choices=CONFIG["models"])
    parser.add_argument("--split", choices=("test_id", "test_ood"))
    args = parser.parse_args()
    if args.command in ("pilot", "train", "evaluate") and not (args.ledger and args.model):
        parser.error("GPU commands require --ledger and --model")
    if args.command in ("pilot", "train") and not args.pretrained:
        parser.error("Training requires --pretrained")
    if args.command == "evaluate" and not args.split:
        parser.error("Evaluation requires --split")
    {"audit": audit, "pilot": train, "train": train, "select": select, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
