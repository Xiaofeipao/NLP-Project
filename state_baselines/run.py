"""Train the three state-policy baselines and freeze development-only selection."""

import argparse
from collections import Counter
from dataclasses import asdict
from functools import partial
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
from state_baselines.prepare_dual_label_data import generation_row, verification_row
from state_baselines.step_verification import step_prompt, verify_steps
from state_baselines.workflow import compare, verifier_report, verify


PROTOCOL = Path(__file__).with_name("protocol.json")
CONFIG = json.loads(PROTOCOL.read_bytes())
STATE_VERIFICATION_WEIGHT = 0.5
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
    result = {
        "source": {path.name: file_hash(path) for path in SOURCES},
        "data": {name: file_hash(args.data / f"{name}.jsonl")
                 for name in ("train", "dev", "test_id", "test_ood")},
        "manifest": file_hash(args.data / "manifest.json"),
        "executor": OfficialOracle(args.executor).sha256,
    }
    if args.step_train:
        result["step_verification"] = {
            "train_sha256": file_hash(args.step_train),
            "scanner_sha256": file_hash(Path(__file__).with_name("step_verification.py")),
        }
    if getattr(args, "dual_train", None):
        result["dual_verification"] = {
            "train_sha256": file_hash(args.dual_train),
            "scanner_sha256": file_hash(Path(__file__).with_name("step_verification.py")),
            "preparation_sha256": file_hash(Path(__file__).with_name("prepare_dual_label_data.py")),
            "state_loss_weight": STATE_VERIFICATION_WEIGHT,
        }
    return result


def training_rows(args):
    rows = read_jsonl(args.data / "train.jsonl")
    if getattr(args, "dual_train", None):
        replacements = read_jsonl(args.dual_train)
        oracle = OfficialOracle(args.executor)
        for source, row in zip(rows, replacements, strict=True):
            grid = Grid.from_dict(source["grid"])
            build = generation_row if source["task"] == "generation" else verification_row
            expected = build(source, grid, oracle)
            if json.dumps(row, sort_keys=True) != json.dumps(expected, sort_keys=True):
                raise ValueError(f"Dual label row does not match its source: {row['id']}")
        return replacements
    if not args.step_train:
        return rows
    replacements = read_jsonl(args.step_train)
    by_parent = {row["parent_id"]: row for row in replacements}
    original = {row["id"]: row for row in rows if row["task"] == "verification"}
    if len(by_parent) != len(replacements) or by_parent.keys() != original.keys():
        raise ValueError("Expected exactly one step verification row per original verification row")
    for key, row in by_parent.items():
        parent = original[key]
        if row["task"] != "verification" or any(
                row[k] != parent[k] for k in ("grid", "layout_id", "source_id", "valid", "reason")):
            raise ValueError(f"Step verification row does not match its source: {row['id']}")
    return [by_parent[row["id"]] if row["task"] == "verification" else row for row in rows]


def state_rows(rows):
    """Generate the V view of a G record without adding rows to the sampler."""
    result = []
    for index, row in enumerate(rows):
        if "state_verification" in row:
            label = row["state_verification"]
            result.append((index, {
                "id": row["id"] + "-state", "task": "verification", "valid": label["valid"],
                "input": step_prompt(Grid.from_dict(row["grid"]), row["policy_position"], None),
                "target": label["local_stop_target"],
            }))
    return result


def collate_dual(runner, pairs):
    primary = runner.collate([feature for feature, _ in pairs])
    states = [state for _, state in pairs if state is not None]
    if states:
        primary["state_batch"] = runner.collate(states)
    return primary


def audit(args):
    training_rows(args)
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


def dev_eval(runner, rows, oracle, check, pilot, verifier=verify):
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
    predictions = verifier(runner, grids_for(verification), [row["candidate"] for row in verification], check)
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
    split_verification_loss = model.get("split_verification_loss", False)
    rows = training_rows(args)
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
        states = state_rows(rows) if args.dual_train else []
        state_samples = [sample for _, sample in states]
        state_features = runner.encode_rows(state_samples) if states else []
        if split_verification_loss:
            verdict_ids = {valid: runner.tokenizer.encode(word, add_special_tokens=False)
                           for valid, word in ((True, "valid"), (False, "invalid"))}
            if any(len(ids) != 1 for ids in verdict_ids.values()) or verdict_ids[True] == verdict_ids[False]:
                raise ValueError("Split verification loss needs distinct single-token verdicts")
            eos = runner.tokenizer.eos_token_id
            for row, feature in zip(rows + state_samples, features + state_features, strict=True):
                if row["task"] == "verification":
                    labels = feature["labels"]
                    if (any(feature["policy_actions"]) or len(labels) < 2
                            or labels[:1] != verdict_ids[row["valid"]] or labels[-1] != eos
                            or (len(labels) == 2) != row["valid"]):
                        raise ValueError(f"Invalid split-verification target: {row['id']}")
        action_index = next(i for i, row in enumerate(rows) if row["task"] == "generation")
        action_batch = runner.collate([features[action_index]]).to(runner.device)
        if action_batch["labels"][0, 0].item() not in runner.action_ids:
            raise ValueError("Action target is misaligned with the first decoder position")
        if action_batch["decoder_input_ids"][0, 0].item() != runner.model.config.decoder_start_token_id:
            raise ValueError("Training and inference use different action decoder starts")
        batch_size = model["microbatch"]
        if states:
            state_by_index = dict(zip((index for index, _ in states), state_features, strict=True))
            dataset = [(feature, state_by_index.get(index)) for index, feature in enumerate(features)]
            collate = partial(collate_dual, runner)
        else:
            dataset, collate = features, runner.collate
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate,
            generator=torch.Generator().manual_seed(CONFIG["seed"]),
        )
        history, best, updates = [], None, 0
        probe = parameters[0].detach().flatten()[:32].clone()
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1, (1 if pilot else CONFIG["epochs"]) + 1):
            runner.model.train()
            total_loss = 0
            total_state_loss, state_examples = 0, 0
            for _, batch, scale, update in accumulated_batches(
                    loader, CONFIG["effective_batch"] // batch_size, "examples"):
                check()
                state_batch = batch.pop("state_batch", None)
                loss = runner.loss(batch.to(runner.device), split_verification_loss=split_verification_loss)
                value = float(loss.detach())
                if not torch.isfinite(loss).item():
                    raise ValueError("Non-finite training loss")
                total_loss += value * len(batch["labels"])
                (loss * scale).backward()
                if state_batch is not None:
                    size = len(state_batch["labels"])
                    state_loss = runner.loss(state_batch.to(runner.device), split_verification_loss=True)
                    state_value = float(state_loss.detach())
                    if not torch.isfinite(state_loss).item():
                        raise ValueError("Non-finite state verification loss")
                    state_scale = STATE_VERIFICATION_WEIGHT * size / len(batch["labels"])
                    (state_loss * scale * state_scale).backward()
                    state_examples += size
                    total_state_loss += state_value * size
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
            metrics, traces = dev_eval(runner, dev, oracle, check, pilot,
                                       verifier=verify_steps if args.step_train or args.dual_train else verify)
            entry = {"epoch": epoch, "train_loss": total_loss / len(rows), **metrics}
            if states:
                if state_examples != len(states):
                    raise ValueError("Not all G state labels participated in the epoch")
                entry.update(state_examples=state_examples, state_loss=total_state_loss / state_examples,
                             objective_loss=(total_loss + STATE_VERIFICATION_WEIGHT * total_state_loss) / len(rows))
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
    names = (args.model,) if args.model else CONFIG["models"]
    results = {name: json.loads((args.out / "train" / name / "training.json").read_bytes())
               for name in names}
    if any(r["identity"] != expected or r["updates"] != 500 for r in results.values()):
        raise ValueError("Selected training runs must be complete and match the current protocol")
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
    candidate_rows = [row for row in rows if "candidate" in row]
    verifier = verify_steps if args.step_train or args.dual_train else verify
    oracle = OfficialOracle(args.executor)
    require_cuda().set_num_threads(1)
    with gpu_budget(args.ledger, 20, f"phase1:{args.split}:{args.model}") as check:
        runner = Seq2Seq(str(checkpoint), CONFIG)
        grids = grids_for(candidate_rows)
        metrics = {"test_status": "reused_exploratory" if args.split != "dev" else "development",
                   "model": args.model}
        if args.split == "dev":
            dev_grids = unique_grids(rows)
            plans = greedy_plans(dev_grids, NeuralPolicy(runner, check), masked=False, max_steps=CONFIG["max_steps"])
            metrics["planning"], traces = compare(runner, dev_grids, plans, oracle, check, verifier=verifier)
            write_jsonl(out / "planning.jsonl", [dict(grid=asdict(grid), **trace)
                                                 for grid, trace in zip(dev_grids, traces, strict=True)])
        elif args.split == "test_id":
            plans = greedy_plans(grids, NeuralPolicy(runner, check), masked=False, max_steps=CONFIG["max_steps"])
            metrics["planning"], traces = compare(runner, grids, plans, oracle, check, verifier=verifier)
            write_jsonl(out / "planning.jsonl", [dict(id=row["id"], grid=row["grid"], **trace)
                                                 for row, trace in zip(rows, traces, strict=True)])
        else:
            metrics["planning"] = {"shared_id_environments": True}
        candidates = [row["candidate"] for row in candidate_rows]
        metrics["fixed_candidates"], traces = compare(runner, grids, candidates, oracle, check, verifier=verifier)
        write_jsonl(out / "candidates.jsonl", [dict(id=row["id"], grid=row["grid"], **trace)
                                               for row, trace in zip(candidate_rows, traces, strict=True)])
        for index, source in enumerate(args.frozen_plans or (), 1):
            saved = read_jsonl(source)
            if [(r["id"], r["grid"]) for r in saved] != [(r["id"], r["grid"]) for r in candidate_rows]:
                raise ValueError(f"Frozen generation inputs do not match the evaluation split: {source}")
            report, traces = compare(runner, grids, [r["direct"] for r in saved], oracle, check,
                                     verifier=verifier)
            name = f"frozen_plans_{index}"
            metrics[name] = {"source": str(source), "sha256": file_hash(source), **report}
            write_jsonl(out / f"{name}.jsonl", [dict(id=r["id"], grid=r["grid"], **trace)
                                                 for r, trace in zip(saved, traces, strict=True)])
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
    parser.add_argument("--split", choices=("dev", "test_id", "test_ood"))
    training_data = parser.add_mutually_exclusive_group()
    training_data.add_argument("--step-train", type=Path)
    training_data.add_argument("--dual-train", type=Path)
    parser.add_argument("--frozen-plans", type=Path, action="append")
    args = parser.parse_args()
    if args.command in ("pilot", "train", "evaluate") and not (args.ledger and args.model):
        parser.error("GPU commands require --ledger and --model")
    if args.command in ("pilot", "train") and not args.pretrained:
        parser.error("Training requires --pretrained")
    if args.command == "evaluate" and not args.split:
        parser.error("Evaluation requires --split")
    if (args.step_train or args.dual_train) and args.command != "audit" and args.model != "flan_t5_base":
        parser.error("Step verification runs require --model flan_t5_base")
    {"audit": audit, "pilot": train, "train": train, "select": select, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
