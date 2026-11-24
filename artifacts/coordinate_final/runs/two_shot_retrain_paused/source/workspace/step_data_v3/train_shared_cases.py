"""Shuffle shared states and train every G target and every V move once per epoch."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

from ppnl.core import ACTIONS
from ppnl.data import read_jsonl
from ppnl.experiment_freeze import file_hash
from ppnl.modeling import require_cuda, gpu_budget
from state_baselines.run import CONFIG
from . import g_probes
from .audit import require
from .build import write_lines, write_object
from .first_token import FirstTokenRunner
from .organize_cases import REVISION, SIZE, WORD, load_states, views
from .path_validation import score_directions
from .single import LABELS
from .train_gpu import assess_dev, split


def case_supervision(cases):
    """G and V each have half the epoch weight. V splits evenly across labels and directions/words."""
    rows, bundles, cells = [], [], Counter()
    for case in cases:
        require(case["state_id"] == case["id"] and [r["candidate"] for r in case["compass"]] == list(ACTIONS), case["id"])
        bundle = []
        for r in views(case):
            require(r["state_id"] == case["state_id"], r["id"])
            bundle.append(len(rows))
            rows.append(r)
            if r["mode"] == "v_step":
                cells[r["target"], r["candidate"]] += 1
        bundles.append(bundle)
    n = len(cases)
    options = Counter(label for label, action in cells)
    require(set(options) == set(LABELS) and all(options[c] == 4 for c in LABELS if c != WORD), "Class-direction cells")
    weights, totals = [], Counter()
    for r in rows:
        if r["mode"] == "g_next":
            weight, key = 0.5, "G"
        else:
            key = r["target"]
            weight = 0.5 * n / (len(LABELS) * options[key] * cells[key, r["candidate"]])
        weights.append(weight)
        totals[key] += weight
    require(math.isclose(totals["G"], n / 2) and all(math.isclose(totals[c], n / (2*len(LABELS))) for c in LABELS),
            "Per-class epoch loss weights")
    return rows, bundles, weights, totals


def report(path, content):
    pending = path.with_suffix(".partial")
    write_object(pending, content)
    pending.replace(path)


def train(args):
    torch = require_cuda()
    torch.set_num_threads(2)
    d, out = args.data, args.out
    manifest = json.loads((d / "manifest.json").read_bytes())
    require(manifest["data_revision"] == REVISION and manifest["template_sha256"] == g_probes.template_hash(), REVISION)
    require(all(file_hash(d / p) == h for p, h in manifest["sha256"].items()), "Dataset checksums")
    require(file_hash(args.pretrained / "tokenizer.json") == manifest["tokenizer_sha256"]
            and (args.pretrained / "upstream_revision.txt").read_text().strip() == CONFIG["models"]["flan_t5_base"]["revision"],
            "Flan base and tokenizer")
    cases = split(d, "train")
    require(len(cases) == len({c["state_id"] for c in cases}) == SIZE, "Shared state budget")
    rows, bundles, weights, totals = case_supervision(cases)
    states = load_states(d)
    dev = split(d, "dev")
    probes = read_jsonl(d / "diagnostics/g_dev_four.jsonl")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with gpu_budget(out / "gpu_ledger.json", 11, "shared-state-gv") as check:
        precision = "bfloat16" if torch.cuda.is_bf16_supported() else "float32"
        settings = dict(CONFIG["training"], precision=precision, eval_batch_size=16, max_input_tokens=512, max_target_tokens=8)
        runner = FirstTokenRunner(str(args.pretrained), dict(CONFIG, training=settings, local_files_only=True))
        runner.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        features = runner.prepare_rows(rows)
        require(len(features) == len(rows) and dict(zip(LABELS, runner.verdict_ids, strict=True))
                == json.loads((d / "audit.json").read_bytes())["verifier_first_tokens"], "Tokenized V targets")
        w_tensor = torch.tensor(weights, device=runner.device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(runner.model.parameters(), lr=settings["learning_rate"],
                                      weight_decay=settings["weight_decay"], fused=True)
        meta = dict(data_manifest=file_hash(d / "manifest.json"), precision=precision, formal_state_cases=len(cases),
                    G_views=len(cases), compass_V=len(ACTIONS)*len(cases), other_V=len(rows)-5*len(cases),
                    total_views=len(rows), g_v_total_epoch_weight=dict(totals), verdict_ids=dict(zip(LABELS, runner.verdict_ids)),
                    objective="0.5 mean G optimal-set NLL + 0.5 mean across five V label losses, balanced by action within label",
                    sampling="seeded shuffle of all states and of all task inputs within each update",
                    seed=CONFIG["seed"], effective_batch_states=args.effective_batch, microbatch_views=args.microbatch,
                    learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"], epochs=args.epochs,
                    endpoint_check=False, device=torch.cuda.get_device_name(), base_revision=CONFIG["models"]["flan_t5_base"]["revision"])
        report(out / "preflight.json", meta)
        generator = torch.Generator().manual_seed(CONFIG["seed"])
        ranks, history, updates = [], [], 0
        for epoch in range(1, args.epochs+1):
            order = torch.randperm(len(cases), generator=generator).tolist()
            view_rng = random.Random(CONFIG["seed"] + epoch - 1)
            visited = bytearray(len(rows))
            epoch_loss, n_views = 0.0, 0
            runner.model.train()
            runner.model.config.use_cache = False
            for offset in range(0, len(order), args.effective_batch):
                group = order[offset:offset+args.effective_batch]
                indices = [i for case_i in group for i in bundles[case_i]]
                view_rng.shuffle(indices)
                optimizer.zero_grad(set_to_none=True)
                loss_sum = torch.zeros((), device=runner.device)
                for j in range(0, len(indices), args.microbatch):
                    check()
                    batch_indices = indices[j:j+args.microbatch]
                    for i in batch_indices:
                        require(not visited[i], "Duplicate case view in epoch")
                        visited[i] = 1
                    batch = runner.collate([features[i] for i in batch_indices])
                    with runner.autocast():
                        losses = runner.batch_losses(batch)
                        weighted = (losses * w_tensor[batch_indices]).sum()
                    loss_sum += weighted.detach()
                    (weighted / len(group)).backward()
                norm = torch.nn.utils.clip_grad_norm_(runner.model.parameters(), 1.0, error_if_nonfinite=True)
                if updates == 0:
                    require(norm.item() > 0, "First gradient")
                optimizer.step()
                updates += 1
                n_views += len(indices)
                epoch_loss += loss_sum.item()
                if updates == 1 or updates % 10 == 0:
                    progress = dict(epoch=epoch, updates=updates, states_seen=offset+len(group), views_seen=n_views,
                                    weighted_mean_loss=epoch_loss/(offset+len(group)), elapsed_seconds=time.monotonic()-started)
                    report(out / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
            require(len(order) == len(set(order)) == SIZE and n_views == len(features) and all(visited), "Incomplete epoch")
            runner.model.eval()
            runner.model.config.use_cache = True
            runner.model.generation_config.use_cache = True
            assessment, records = assess_dev(runner, dev, states, check)
            write_lines(out / f"dev_steps_epoch_{epoch}.jsonl", records)
            directions = score_directions(runner, probes, states, out / f"dev_directions_epoch_{epoch}.jsonl", goal_check=False)
            assessment.update(epoch=epoch, directions=directions, updates=updates, views=n_views, states=len(order),
                              train_mean_loss=epoch_loss/len(order), state_order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest())
            history.append(assessment)
            checkpoint = out / f"checkpoint-epoch-{epoch}"
            runner.model.save_pretrained(checkpoint)
            runner.tokenizer.save_pretrained(checkpoint)
            ranks.append((assessment["V"]["macro_f1"], assessment["V"]["correct"], -epoch))
            best_epoch = -max(ranks)[-1]
            report(out / "selection.json", dict(selected=f"checkpoint-epoch-{best_epoch}", epoch=best_epoch,
                   rank=["dev_step_macro_f1","dev_step_correct","earlier_epoch"]))
            report(out / "train_history.json", history)
            print(json.dumps({"epoch":epoch, "step_macro_f1":assessment["V"]["macro_f1"], "best":best_epoch}), flush=True)
        report(out / "train_result.json", dict(experiment=meta, updates=updates, checkpoint=f"checkpoint-epoch-{best_epoch}",
               seconds=time.monotonic()-started, dev=history))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data","pretrained","out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--epochs", type=int, choices=(1,2,3), default=3)
    parser.add_argument("--effective-batch", type=int, default=32)
    parser.add_argument("--microbatch", type=int, default=4)
    args = parser.parse_args()
    if min(args.effective_batch, args.microbatch) < 1:
        parser.error("Batch sizes must be positive")
    train(args)
