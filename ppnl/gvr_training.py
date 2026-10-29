"""Fixed-subset G/V/R training; no test data enters model selection."""
from __future__ import annotations

import json
from pathlib import Path
import time

from .data import task_index, task_index_manifest, write_json
from .modeling import Seq2Seq, confirmation, gpu_budget, require_cuda
from .task_adapters import attach_adapter, checkpoint_hash


def train_stage(config, protocol, stage, output, pilot=False):
    if stage not in ("generation", "verification", "repair"):
        raise ValueError(stage)
    evidence = protocol["evidence"]
    if not all(evidence.get(k) for k in ("gpu_ledger_reconciled", "checkpoint_manager_confirmed", "baseline_outputs_verified")):
        raise ValueError("P0 evidence is incomplete; do not start GPU training")
    item = confirmation(config, "flan_t5_small")
    if not item.get("manager_confirmed"):
        raise ValueError("Project manager checkpoint confirmation is required for GVR training")
    torch = require_cuda()
    root = Path(protocol["dataset_root"])
    index = task_index_manifest(root)
    expected = json.loads(Path(protocol["task_index"]).read_text(encoding="utf-8"))
    if index != expected:
        raise ValueError("Frozen task index changed")
    if not pilot:
        pilot_file = Path(protocol["output_root"]) / stage / "pilot" / "training.json"
        if not pilot_file.exists():
            raise ValueError("Run and inspect this stage's pilot first")
        prior = json.loads(pilot_file.read_text(encoding="utf-8"))
        if (prior["stage"] != stage or not prior["pilot"] or
                prior["task_index_source_sha256"] != index["source_manifest_sha256"] or
                prior["training_config"] != config["training"]):
            raise ValueError("Pilot protocol changed")
        if stage == "generation" and (prior["source_checkpoint"] != item["checkpoint"] or
                                      prior["source_revision"] != item.get("revision")):
            raise ValueError("Pretrained checkpoint changed since pilot")
    base = Path(protocol["generation_base"])
    if stage != "generation" and not (base / "config.json").exists():
        raise ValueError("Train and freeze G first")
    base_hash = checkpoint_hash(base) if stage != "generation" else None
    if not pilot and prior["generation_base_sha256"] != base_hash:
        raise ValueError("Generation base changed since pilot")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh training output directory")
    rows, dev = task_index(root, "train", stage), task_index(root, "dev", stage)
    settings = config["training"]
    epochs = 1 if pilot else settings["epochs"]
    if not 1 <= epochs <= 2:
        raise ValueError("Maximum two formal epochs")
    if pilot:
        rows = rows[:settings["batch_size"] * settings["pilot_batches"]]
    out.mkdir(parents=True, exist_ok=True)
    with gpu_budget(config["gpu_ledger"], config["gpu_hours_cap"], f"gvr:{stage}:{'pilot' if pilot else 'train'}") as check:
        runner = Seq2Seq(item["checkpoint"] if stage == "generation" else str(base), config,
                         item.get("revision") if stage == "generation" else None)
        if stage != "generation":
            adapter_info = attach_adapter(runner, stage, protocol["lora"])
        else:
            adapter_info = None
        lengths = runner.check_lengths(rows + dev)
        generator = torch.Generator().manual_seed(config["seed"])
        loader = torch.utils.data.DataLoader(runner.encode_rows(rows), batch_size=settings["batch_size"], shuffle=True,
                                             generator=generator, collate_fn=runner.collate, pin_memory=True)
        dev_loader = torch.utils.data.DataLoader(runner.encode_rows(dev), batch_size=settings["eval_batch_size"],
                                                 collate_fn=runner.collate, pin_memory=True)
        optimizer = torch.optim.AdamW((p for p in runner.model.parameters() if p.requires_grad),
                                      lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        best, history, processed, updates = float("inf"), [], 0, 0
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for epoch in range(epochs):
            runner.model.train()
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(loader):
                check()
                batch = batch.to(runner.device, non_blocking=True)
                accumulation = settings["gradient_accumulation_steps"]
                window = min(accumulation, len(loader) - step // accumulation * accumulation)
                with runner.autocast():
                    loss = runner.loss(batch)
                (loss / window).backward()
                processed += len(batch["input_ids"])
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    torch.nn.utils.clip_grad_norm_(runner.model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
            runner.model.eval()
            weighted_loss = tokens = 0
            with torch.inference_mode():
                for batch in dev_loader:
                    check()
                    batch = batch.to(runner.device, non_blocking=True)
                    n = (batch["labels"] != -100).sum().item()
                    with runner.autocast():
                        weighted_loss += runner.loss(batch).item() * n
                    tokens += n
            dev_loss = weighted_loss / tokens
            history.append({"epoch": epoch + 1, "dev_loss": dev_loss})
            if dev_loss < best:
                best = dev_loss
                if not pilot:
                    runner.model.save_pretrained(out / "best")
                    if stage == "generation":
                        runner.tokenizer.save_pretrained(out / "best")
        torch.cuda.synchronize()
        if base_hash and checkpoint_hash(base) != base_hash:
            raise RuntimeError("Frozen G changed during adapter training")
        report = {"stage": stage, "pilot": pilot, "train_rows": len(rows), "dev_rows": len(dev),
                  "examples_processed": processed, "optimizer_updates": updates, "history": history,
                  "source_checkpoint": item["checkpoint"] if stage == "generation" else str(base),
                  "source_revision": runner.source_revision, "generation_base_sha256": base_hash,
                  "task_index_source_sha256": index["source_manifest_sha256"], "lengths": lengths,
                  "training_config": settings,
                  "adapter": adapter_info, "gpu": torch.cuda.get_device_name(),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "elapsed_seconds": time.perf_counter() - started}
        write_json(out / "training.json", report)
        if not pilot and stage != "generation":
            write_json(out / "best" / stage / "base_identity.json", {"generation_base_sha256": base_hash,
                       "stage": stage, "task_index_source_sha256": index["source_manifest_sha256"]})
        return report
