"""Train and evaluate the balanced single-step G/V dataset on a CUDA device."""
import argparse
from collections import Counter
from pathlib import Path
import json
import time

from ppnl.core import Grid
from ppnl.data import read_jsonl, write_json, write_jsonl
from ppnl.evaluation import plan_metrics
from ppnl.experiment_freeze import file_hash
from ppnl.modeling import Seq2Seq, gpu_budget, require_cuda
from ppnl.optimization import accumulated_batches
from ppnl.policy import NAMES, greedy_plans
from state_baselines.run import CONFIG
from state_baselines.workflow import verifier_report
from . import g_probes
from .model_io import read_split
from .schema import State
from .shortcut_audit import metrics
from .single import LABELS


def split(dataset, name):
    return read_split(dataset, name, version=g_probes.VERSION, fingerprint=g_probes.template_hash())


def validate_inputs(dataset, pretrained):
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    revision = (pretrained / "upstream_revision.txt").read_text().strip()
    if revision != CONFIG["models"]["flan_t5_base"]["revision"]:
        raise ValueError("Unexpected pretrained revision")
    if file_hash(pretrained / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Unexpected pretrained tokenizer")
    for name, expected in manifest["sha256"].items():
        if file_hash(dataset / name) != expected:
            raise ValueError(f"Changed dataset file: {name}")
    states = {r["id"]: State.from_record(r) for r in read_jsonl(dataset / "states.jsonl")}
    environments = {name: {states[r["state_id"]].record()["layout_id"] for r in split(dataset, name)}
                    for name in ("train", "dev")}
    for name in ("test_id", "test_ood"):
        environments[name] = {r["layout_id"] for r in split(dataset, name)}
    for a, b in (("train", "dev"), ("train", "test_id"), ("train", "test_ood"),
                 ("dev", "test_id"), ("dev", "test_ood")):
        if not environments[a].isdisjoint(environments[b]):
            raise ValueError(f"Layout leakage between {a} and {b}")
    return states


def local_v(runner, rows, check):
    predictions = [p.strip() for p in runner.predict([r["input"] for r in rows], check)]
    result = metrics([r["target"] for r in rows], predictions)
    result["malformed"] = sum(p not in LABELS for p in predictions)
    return result, [dict(id=r["id"], target=r["target"], prediction=p)
                    for r, p in zip(rows, predictions, strict=True)]


def assess_dev(runner, rows, states, check):
    gs = [r for r in rows if r["mode"] == "g_next"]
    vs = [r for r in rows if r["mode"] == "v_step"]
    grids = [states[r["state_id"]].grid() for r in gs]
    scorer = g_probes.generation_policy(runner, check)
    scores = scorer.scores([(g, g.start) for g in grids])
    correct = sum(NAMES[max(range(4), key=s.__getitem__)] in r["optimal_actions"]
                  for r, s in zip(gs, scores, strict=True))
    plans = greedy_plans(grids, scorer, masked=False, max_steps=CONFIG["max_steps"])
    verification, records = local_v(runner, vs, check)
    return {"G_one_step_correct": correct, "G_one_step_total": len(gs),
            "G_greedy": plan_metrics(grids, plans), "V": verification}, records


def assess_test(runner, dataset, name, out, check):
    rows = split(dataset, name)
    grids = [Grid.from_dict(r["grid"]) for r in rows]
    predictions = g_probes.verify_plans(runner, grids, [r["candidate"] for r in rows], check)
    report = verifier_report(rows, predictions)
    report["exact"] = sum(r["label"] == p for r, p in zip(rows, predictions, strict=True))
    write_jsonl(out / f"{name}_paths.jsonl",
                [dict(id=r["id"], target=r["label"], prediction=p)
                 for r, p in zip(rows, predictions, strict=True)])
    local, local_records = local_v(runner, read_jsonl(dataset / "diagnostics" / f"{name}_local.jsonl"), check)
    balanced, _ = local_v(runner, g_probes.read_heldout_subset(dataset, name), check)
    write_jsonl(out / f"{name}_steps.jsonl", local_records)
    return {"full_paths": report, "local_steps": local, "balanced_local_steps": balanced}


def train(args, states, phase, microbatch, precision, check):
    from transformers import AutoModelForSeq2SeqLM

    torch = require_cuda()
    torch.set_num_threads(2)
    settings = dict(CONFIG["training"], precision=precision, eval_batch_size=16,
                    max_input_tokens=128, max_target_tokens=8)
    runner = Seq2Seq(str(args.pretrained), dict(CONFIG, training=settings))
    runner.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    features = g_probes.prepare_training(runner, args.data)
    dev = split(args.data, "dev")
    if phase == "pilot":
        rows = split(args.data, "train")
        by_task = [[i for i, r in enumerate(rows) if r["mode"] == mode][:16] for mode in ("g_next", "v_step")]
        features = [features[i] for indices in by_task for i in indices]
        dev = [r for mode in ("g_next", "v_step") for r in [r for r in dev if r["mode"] == mode][:12]]
    loader = torch.utils.data.DataLoader(features, batch_size=microbatch, shuffle=True, collate_fn=runner.collate,
                                        generator=torch.Generator().manual_seed(CONFIG["seed"]), pin_memory=True)
    optimizer = torch.optim.AdamW(runner.model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"], fused=True)
    torch.cuda.reset_peak_memory_stats()
    history, best, best_rank = [], None, None
    updates = 0
    started = time.monotonic()
    for epoch in range(1, 2 if phase == "pilot" else CONFIG["epochs"] + 1):
        runner.model.train()
        runner.model.config.use_cache = False
        optimizer.zero_grad(set_to_none=True)
        loss_sum, seen = 0.0, 0
        for _, batch, scale, update in accumulated_batches(loader, CONFIG["effective_batch"] // microbatch, "examples"):
            check()
            with runner.autocast():
                loss = g_probes.batch_loss(runner, batch)
            (loss * scale).backward()
            loss_sum += loss.detach().float().item() * len(batch["labels"])
            seen += len(batch["labels"])
            if update:
                norm = torch.nn.utils.clip_grad_norm_(runner.model.parameters(), 1.0, error_if_nonfinite=True)
                if updates == 0 and not norm.item() > 0:
                    raise ValueError("First optimizer update has no gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                if updates % 25 == 0 or phase == "pilot":
                    progress = {"phase": phase, "epoch": epoch, "updates": updates, "epoch_examples": seen,
                                "train_mean_loss": loss_sum / seen, "elapsed_seconds": time.monotonic() - started,
                                "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20)}
                    write_json(args.out / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
        runner.model.eval()
        runner.model.config.use_cache = True
        runner.model.generation_config.use_cache = True
        assessment, records = assess_dev(runner, dev, states, check)
        assessment.update(epoch=epoch, updates=updates, train_mean_loss=loss_sum / seen)
        history.append(assessment)
        write_json(args.out / f"{phase}_history.json", history)
        write_jsonl(args.out / f"{phase}_dev_epoch_{epoch}.jsonl", records)
        print(json.dumps(assessment), flush=True)
        rank = (assessment["V"]["macro_f1"], assessment["G_greedy"]["success_rate"], assessment["G_one_step_correct"])
        if phase == "train" and (best_rank is None or rank > best_rank):
            best_rank = rank
            best = args.out / f"checkpoint-epoch-{epoch}"
            runner.model.save_pretrained(best)
            runner.tokenizer.save_pretrained(best)
            write_json(args.out / "selection.json", {"selected": best.name, "epoch": epoch,
                       "rank": ["dev V macro F1", "dev G greedy success rate", "dev G one-step correct"]})
    result = {"phase": phase, "microbatch": microbatch, "precision": precision, "updates": updates,
              "gpu": torch.cuda.get_device_name(), "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
              "seconds": time.monotonic() - started, "dev": history}
    if best:
        del optimizer
        del runner.model
        torch.cuda.empty_cache()
        runner.model = AutoModelForSeq2SeqLM.from_pretrained(best, local_files_only=True).to(runner.device)
        runner.model.eval()
        result["checkpoint"] = best.name
        result["test"] = {name: assess_test(runner, args.data, name, args.out, check)
                          for name in ("test_id", "test_ood")}
    write_json(args.out / f"{phase}_result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=4, choices=(1, 2, 4))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    states = validate_inputs(args.data, args.pretrained)
    torch = require_cuda()
    precision = "bfloat16" if torch.cuda.is_bf16_supported() else "float32"
    microbatch = args.microbatch
    with gpu_budget(args.out / "gpu_ledger.json", 20, "balanced-gv") as check:
        while True:
            try:
                print(f"Pilot: microbatch={microbatch}, precision={precision}", flush=True)
                train(args, states, "pilot", microbatch, precision, check)
                break
            except torch.cuda.OutOfMemoryError:
                if microbatch == 1:
                    raise
            microbatch //= 2
            torch.cuda.empty_cache()
        train(args, states, "train", microbatch, precision, check)


if __name__ == "__main__":
    main()
