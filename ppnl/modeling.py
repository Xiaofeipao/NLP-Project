"""Lazy ML imports: data utilities do not require PyTorch or a GPU."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import time

from .data import read_jsonl, write_json
from .optimization import ACCUMULATION_NORMALIZATION, accumulated_batches


@contextmanager
def gpu_budget(path, cap_hours, operation):
    """Wall-clock GPU reservation, including load and validation. Run jobs serially."""
    path = Path(path)
    ledger = json.loads(path.read_text()) if path.exists() else {"runs": []}
    used = sum(r["seconds"] for r in ledger["runs"])
    remaining = cap_hours * 3600 - used
    if remaining <= 0:
        raise RuntimeError("GPU-hour allocation exhausted")
    start = time.monotonic()

    def check():
        if time.monotonic() - start >= remaining:
            raise RuntimeError("GPU-hour allocation reached; stopping at batch boundary")

    status = "failed"
    try:
        yield check
        status = "completed"
    finally:
        ledger["runs"].append({"operation": operation, "seconds": time.monotonic() - start,
                               "status": status, "started_unix": time.time() - (time.monotonic() - start)})
        write_json(path, ledger)


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for model commands. Data preparation works on CPU.")
    return torch


def confirmation(config, key):
    item = config["models"][key]
    if not (item.get("manager_confirmed") or item.get("user_confirmed")) or not item.get("confirmation_note", "").strip():
        raise ValueError(f"Confirm exact checkpoint with project manager first: {item['checkpoint']}. "
                         "Set manager_confirmed and confirmation_note in config.json.")
    return item


def action_set_loss(logits, labels, action_ids, allowed, mode):
    """Per-example objective: four-action CE/set mass for policy, text CE for verifier."""
    import torch
    import torch.nn.functional as F
    is_policy = allowed.any(dim=1)
    action_logits = logits[:, 0, action_ids].float()
    if mode == "single_action":
        ids = torch.tensor(action_ids, device=labels.device)
        targets = (labels[:, :1] == ids[None, :]).long().argmax(dim=1)
        policy_loss = F.cross_entropy(action_logits, targets, reduction="none")
    elif mode == "optimal_actions":
        # Avoid all-minus-infinity rows and NaN gradients for verification examples.
        safe_allowed = torch.where(is_policy[:, None], allowed, torch.ones_like(allowed))
        probabilities = F.log_softmax(action_logits, dim=-1)
        policy_loss = -torch.logsumexp(probabilities.masked_fill(~safe_allowed, -float("inf")), dim=-1)
    else:
        raise ValueError(mode)
    tokens = F.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
    text_loss = tokens.sum(dim=1) / (labels != -100).sum(dim=1).clamp_min(1)
    return torch.where(is_policy, policy_loss, text_loss).mean()


class Seq2Seq:
    def __init__(self, checkpoint, config, revision=None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, set_seed
        set_seed(config["seed"])
        self.config, self.torch = config, torch
        self.device = "cuda"
        self.precision = config["training"].get("precision", "float32")
        self.policy_loss = config["training"].get("policy_loss")
        if self.precision not in ("float32", "bfloat16"):
            raise ValueError("precision must be float32 or bfloat16")
        if self.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support BF16")
        torch.set_float32_matmul_precision("high")
        kwargs = {"cache_dir": config.get("model_cache", ".cache/huggingface"),
                  "local_files_only": config.get("local_files_only", False)}
        if revision:
            kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, **kwargs)
        self.action_ids = None
        if self.policy_loss:
            token_ids = [self.tokenizer.encode(a, add_special_tokens=False) for a in ("up", "down", "left", "right")]
            if any(len(ids) != 1 for ids in token_ids):
                raise ValueError("Action loss requires single-token action words")
            self.action_ids = [ids[0] for ids in token_ids]
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, **kwargs).to(self.device)
        self.source_revision = getattr(self.model.config, "_commit_hash", None) or revision
        # Upstream BART config enables beam-only early stopping; use a clean greedy setting.
        self.model.generation_config.early_stopping = False
        self.model.generation_config.no_repeat_ngram_size = 0
        self._collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model, pad_to_multiple_of=8)
        self.input_limit = config["training"]["max_input_tokens"]
        self.output_limit = config["training"]["max_target_tokens"]
        positional = getattr(self.model.config, "max_position_embeddings", self.input_limit)
        if self.input_limit > positional:
            raise ValueError(f"Input limit {self.input_limit} exceeds model positions {positional}")

    def check_lengths(self, rows):
        lengths, targets = [], []
        for row in rows:
            n = len(self.tokenizer(row["input"])["input_ids"])
            m = len(self.tokenizer(text_target=row["target"])["input_ids"])
            if n > self.input_limit or m > self.output_limit:
                raise ValueError(f"{row['id']}: input={n}, target={m}; truncation is forbidden")
            lengths.append(n)
            targets.append(m)
        return {"max_input_tokens": max(lengths), "max_target_tokens": max(targets)}

    def autocast(self):
        return (self.torch.autocast("cuda", dtype=self.torch.bfloat16)
                if self.precision == "bfloat16" else nullcontext())

    def encode_rows(self, rows):
        """Tokenize once, on CPU, and reuse encodings across epochs."""
        inputs = self.tokenizer([r["input"] for r in rows], truncation=False)
        targets = self.tokenizer(text_target=[r["target"] for r in rows], truncation=False)
        features = []
        for i, row in enumerate(rows):
            n, m = len(inputs["input_ids"][i]), len(targets["input_ids"][i])
            if n > self.input_limit or m > self.output_limit:
                raise ValueError(f"{row['id']}: input={n}, target={m}; truncation is forbidden")
            features.append({k: values[i] for k, values in inputs.items()})
            features[-1]["labels"] = targets["input_ids"][i]
            if self.policy_loss:
                actions = row.get("optimal_actions", [])
                features[-1]["policy_actions"] = [a in actions for a in ("up", "down", "left", "right")]
        return features

    def collate(self, rows):
        features = self.encode_rows(rows) if "input" in rows[0] else rows
        if not self.policy_loss:
            return self._collator(features)
        copied = [dict(f) for f in features]
        masks = [f.pop("policy_actions") for f in copied]
        batch = self._collator(copied)
        batch["policy_actions"] = self.torch.tensor(masks, dtype=self.torch.bool)
        return batch

    def loss(self, batch):
        inputs = {k: v for k, v in batch.items() if k != "policy_actions"}
        output = self.model(**inputs)
        if self.policy_loss:
            return action_set_loss(output.logits, batch["labels"], self.action_ids,
                                   batch["policy_actions"], self.policy_loss)
        return output.loss

    def predict(self, inputs, check=lambda: None):
        torch = self.torch
        result = []
        self.model.eval()
        bs = self.config["training"]["eval_batch_size"]
        for offset in range(0, len(inputs), bs):
            check()
            strings = inputs[offset:offset + bs]
            encoded = self.tokenizer(strings, padding=True, pad_to_multiple_of=8,
                                     truncation=False, return_tensors="pt")
            if encoded.input_ids.shape[1] > self.input_limit:
                raise ValueError("Inference input exceeds configured limit; refusing truncation")
            with torch.inference_mode(), self.autocast():
                output = self.model.generate(**encoded.to(self.device), do_sample=False, num_beams=1,
                                             no_repeat_ngram_size=0, repetition_penalty=1.0,
                                             max_new_tokens=self.output_limit)
            # Keep model output strict; do not strip non-action prose or punctuation.
            result.extend(self.tokenizer.batch_decode(output, skip_special_tokens=True))
        return result


def dataset_fingerprint(directory):
    value = hashlib.sha256()
    for name in ("manifest.json", "train.jsonl", "dev.jsonl"):
        value.update((Path(directory) / name).read_bytes())
    return value.hexdigest()


def train(config, key, dataset, output, pilot=False):
    item = confirmation(config, key)
    torch = require_cuda()
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose a fresh output directory: {out}")
    settings = config["training"]
    if not 1 <= settings["epochs"] <= 2:
        raise ValueError("Use 1 or 2 epochs: reserve at most one additional pass for the pilot")
    train_rows = read_jsonl(Path(dataset) / "train.jsonl")
    dev_rows = read_jsonl(Path(dataset) / "dev.jsonl")
    pilot_path = Path(config["output_root"]) / key / "pilot" / "pilot.json"
    fingerprint = dataset_fingerprint(dataset)
    revision = item.get("revision")
    if not pilot:
        if not pilot_path.exists():
            raise ValueError("Run and inspect the pilot first")
        previous = json.loads(pilot_path.read_text())
        if previous["checkpoint"] != item["checkpoint"] or previous["data_fingerprint"] != fingerprint:
            raise ValueError("Pilot checkpoint/data mismatch")
        if previous["training_config"] != settings:
            raise ValueError("Pilot training configuration changed; review before proceeding")
        if previous.get("accumulation_normalization") != ACCUMULATION_NORMALIZATION:
            raise ValueError("Pilot accumulation normalization changed; run a fresh pilot")
        revision = previous["resolved_revision"] or revision
    out.mkdir(parents=True, exist_ok=True)
    with gpu_budget(config["gpu_ledger"], config["gpu_hours_cap"], f"{'pilot' if pilot else 'train'}:{key}") as check:
        runner = Seq2Seq(item["checkpoint"], config, revision)
        lengths = runner.check_lengths(train_rows + dev_rows)
        torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator().manual_seed(config["seed"])
        if pilot:
            # One partial pass only; baseline training gets at most two further passes.
            train_rows = train_rows[:settings["batch_size"] * settings["pilot_batches"]]
        loader = torch.utils.data.DataLoader(runner.encode_rows(train_rows), batch_size=settings["batch_size"],
                                             shuffle=True, generator=generator, collate_fn=runner.collate,
                                             num_workers=0, pin_memory=True)
        dev_loader = torch.utils.data.DataLoader(runner.encode_rows(dev_rows), batch_size=settings["eval_batch_size"],
                                                 collate_fn=runner.collate, num_workers=0, pin_memory=True)
        optimizer = torch.optim.AdamW(runner.model.parameters(), lr=settings["learning_rate"],
                                      weight_decay=settings["weight_decay"])
        epochs = 1 if pilot else settings["epochs"]
        history, best, processed, updates = [], float("inf"), 0, 0
        torch.cuda.synchronize()
        start = time.perf_counter()
        for epoch in range(epochs):
            print(f"{key}: epoch {epoch + 1}/{epochs}, {len(loader)} batches", flush=True)
            runner.model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            reduction = "examples" if runner.policy_loss else "tokens"
            for step, batch, loss_scale, update in accumulated_batches(
                    loader, settings["gradient_accumulation_steps"], reduction):
                check()
                batch = batch.to(runner.device, non_blocking=True)
                with runner.autocast():
                    loss = runner.loss(batch)
                (loss * loss_scale).backward()
                epoch_loss += loss.detach().item()
                processed += batch["input_ids"].shape[0]
                if update:
                    torch.nn.utils.clip_grad_norm_(runner.model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                if (step + 1) % 100 == 0 or step + 1 == len(loader):
                    progress = {"model": key, "epoch": epoch + 1, "epochs": epochs,
                                "batch": step + 1, "batches": len(loader), "updates": updates,
                                "mean_train_loss": epoch_loss / (step + 1),
                                "elapsed_seconds": time.perf_counter() - start}
                    write_json(out / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
            runner.model.eval()
            total_loss, ntokens = 0.0, 0
            with torch.inference_mode():
                for batch in dev_loader:
                    check()
                    batch = batch.to(runner.device, non_blocking=True)
                    n = batch["labels"].shape[0] if runner.policy_loss else (batch["labels"] != -100).sum().item()
                    with runner.autocast():
                        total_loss += runner.loss(batch).item() * n
                    ntokens += n
            dev_loss = total_loss / ntokens
            history.append({"epoch": epoch + 1, "train_batch_mean_loss": epoch_loss / len(loader),
                            "dev_loss": dev_loss, "dev_token_mean_loss": None if runner.policy_loss else dev_loss})
            print(f"{key}: epoch {epoch + 1}, dev loss {dev_loss:.6f}", flush=True)
            if dev_loss < best:
                best = dev_loss
                if not pilot:
                    runner.model.save_pretrained(out / "best")
                    runner.tokenizer.save_pretrained(out / "best")
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - start
        # Report exact match for the saved best checkpoint, not a potentially worse last epoch.
        if not pilot:
            from transformers import AutoModelForSeq2SeqLM
            del optimizer
            del runner.model
            torch.cuda.empty_cache()
            runner.model = AutoModelForSeq2SeqLM.from_pretrained(out / "best", local_files_only=True).to("cuda")
            runner._collator.model = runner.model
            runner.model.generation_config.early_stopping = False
            runner.model.generation_config.no_repeat_ngram_size = 0
        eval_start = time.perf_counter()
        policy_accuracy = None
        if runner.policy_loss:
            from .core import Grid
            from .policy import NeuralPolicy, NAMES
            predictions = [None] * len(dev_rows)
            policy_indices = [i for i, r in enumerate(dev_rows) if r.get("optimal_actions")]
            text_indices = [i for i, r in enumerate(dev_rows) if not r.get("optimal_actions")]
            scores = NeuralPolicy(runner, check).scores([(Grid.from_dict(dev_rows[i]["grid"]), tuple(dev_rows[i]["policy_position"]))
                                                       for i in policy_indices])
            for i, values in zip(policy_indices, scores):
                predictions[i] = NAMES[max(range(4), key=lambda j: values[j])]
            for i, value in zip(text_indices, runner.predict([dev_rows[i]["input"] for i in text_indices], check)):
                predictions[i] = value
            policy_accuracy = sum(predictions[i] in dev_rows[i]["optimal_actions"] for i in policy_indices) / len(policy_indices)
        else:
            predictions = runner.predict([r["input"] for r in dev_rows], check)
        torch.cuda.synchronize()
        eval_seconds = time.perf_counter() - eval_start
        report = {"checkpoint": item["checkpoint"], "requested_revision": item.get("revision"),
                  "resolved_revision": runner.source_revision,
                  "data_fingerprint": fingerprint, "training_config": settings,
                  "accumulation_normalization": ACCUMULATION_NORMALIZATION,
                  "confirmation_note": item["confirmation_note"],
                  "gpu": torch.cuda.get_device_name(), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "lengths": lengths,
                  "examples_processed": processed, "optimizer_updates": updates,
                  "train_and_loss_eval_seconds": train_seconds,
                  "train_examples_per_second_including_validation": processed / train_seconds,
                  "generation_eval_seconds": eval_seconds,
                  "generation_examples_per_second": len(dev_rows) / eval_seconds, "history": history,
                  "dev_exact_match": sum(a.strip() == b["target"] for a, b in zip(predictions, dev_rows)) / len(dev_rows),
                  "dev_policy_action_accuracy": policy_accuracy,
                  "torch_version": torch.__version__, "precision": runner.precision,
                  "loss_normalization": "example_mean_action_classification_and_verifier_text" if runner.policy_loss else "token_mean_text",
                  "matmul_precision": torch.get_float32_matmul_precision()}
        write_json(out / ("pilot.json" if pilot else "training.json"), report)
    return report


def smoke(config, key, dataset, output):
    """A bounded technical check, never a formal model-selection result.

    No manager confirmation is fabricated. Touches only training and development
    data and writes neither a trained checkpoint nor a model-selection score.
    """
    import sys
    import transformers
    from .prompts import prompt
    from .core import Grid, OfficialOracle
    from .evaluation import evaluate
    torch = require_cuda()
    item = config["models"][key]
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose a fresh smoke output directory: {out}")
    train_rows = read_jsonl(Path(dataset) / "train.jsonl")
    dev_rows = read_jsonl(Path(dataset) / "dev.jsonl")
    # A single small pass, balanced across all three tasks.
    sample = [r for task in ("generation", "verification", "repair")
              for r in [r for r in train_rows if r["task"] == task][:8]]
    check_rows = [next(r for r in dev_rows if r["task"] == task)
                  for task in ("generation", "verification", "repair")]
    with gpu_budget(config["gpu_ledger"], config["gpu_hours_cap"], f"smoke:{key}") as check:
        runner = Seq2Seq(item["checkpoint"], config, item.get("revision"))
        lengths = runner.check_lengths(train_rows + dev_rows)
        features = runner.encode_rows(sample)
        batches = [runner.collate(features[i:i + 8]).to("cuda") for i in range(0, len(features), 8)]
        optimizer = torch.optim.AdamW(runner.model.parameters(), lr=config["training"]["learning_rate"])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        runner.model.eval()
        with torch.inference_mode():
            before = []
            for batch in batches:
                with runner.autocast():
                    before.append(runner.model(**batch).loss.item())
        start = time.perf_counter()
        losses = []
        runner.model.train()
        for i, batch in enumerate(batches):
            check()
            optimizer.zero_grad(set_to_none=True)
            with runner.autocast():
                loss = runner.model(**batch).loss
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(runner.model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append({"step": i + 1, "loss": loss.item(), "gradient_norm": norm.item()})
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - start
        runner.model.eval()
        with torch.inference_mode():
            after = []
            for batch in batches:
                with runner.autocast():
                    after.append(runner.model(**batch).loss.item())
        start = time.perf_counter()
        predictions = runner.predict([r["input"] for r in check_rows], check)
        torch.cuda.synchronize()
        eval_seconds = time.perf_counter() - start
        # Exercise real generate -> verify -> repair inputs without touching test sets.
        grid = Grid.from_dict(check_rows[0]["grid"])
        generated = runner.predict([prompt("generation", grid)], check)[0]
        feedback = runner.predict([prompt("verification", grid, generated)], check)[0]
        repaired = runner.predict([prompt("repair", grid, generated, feedback)], check)[0]
        # Validate the complete evaluator on development candidates only.
        manifest = json.loads((Path(dataset) / "manifest.json").read_text())
        oracle = OfficialOracle(manifest["executor_path"])
        if oracle.sha256 != manifest["executor_sha256"]:
            raise ValueError("Official executor changed")
        eval_rows = []
        for valid in (True, False):
            eval_rows.extend([r for r in dev_rows if r["task"] == "verification" and r["valid"] == valid][:2])
        print(f"{key}: training steps complete; checking full pipeline on {len(eval_rows)} dev candidates", flush=True)
        pipeline_metrics = evaluate(runner, eval_rows, out / "dev_pipeline", check, full=True, rounds=2, oracle=oracle)
        # Same resident model and same batch: no optimizer updates in this microbenchmark.
        timings = {}
        selected_precision = runner.precision
        run_peak_allocated = torch.cuda.max_memory_allocated()
        run_peak_reserved = torch.cuda.max_memory_reserved()
        for precision in ("float32", "bfloat16"):
            check()
            runner.precision = precision
            runner.model.eval()
            for trial in range(4):
                optimizer.zero_grad(set_to_none=True)
                if trial == 1:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    bench_start = time.perf_counter()
                with runner.autocast():
                    loss = runner.model(**batches[0]).loss
                loss.backward()
            torch.cuda.synchronize()
            timings[precision] = {"forward_backward_seconds_per_batch": (time.perf_counter() - bench_start) / 3,
                                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "loss": loss.item()}
        optimizer.zero_grad(set_to_none=True)
        runner.precision = selected_precision
        report = {"purpose": "technical_smoke_only_not_formal_baseline", "model": key,
                  "checkpoint": item["checkpoint"], "resolved_revision": getattr(runner.model.config, "_commit_hash", None),
                  "python_executable": sys.executable, "torch": torch.__version__,
                  "transformers": transformers.__version__, "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(), "precision": runner.precision,
                  "matmul_precision": torch.get_float32_matmul_precision(),
                  "lengths": lengths, "training_examples": len(sample), "optimizer_steps": len(batches),
                  "training_seconds": train_seconds, "training_examples_per_second": len(sample) / train_seconds,
                  "before_loss": sum(before) / len(before), "after_loss": sum(after) / len(after),
                  "steps": losses, "peak_allocated_bytes": max(run_peak_allocated, *(t["peak_allocated_bytes"] for t in timings.values())),
                  "peak_reserved_bytes": max(run_peak_reserved, torch.cuda.max_memory_reserved()), "generation_seconds": eval_seconds,
                  "previews": [{"task": r["task"], "id": r["id"], "target": r["target"], "prediction": p}
                               for r, p in zip(check_rows, predictions)],
                  "pipeline": {"generated": generated, "self_feedback": feedback, "repaired": repaired},
                  "dev_pipeline_metrics": pipeline_metrics,
                  "precision_microbenchmark": timings,
                  "data_fingerprint": dataset_fingerprint(dataset)}
        write_json(out / "smoke.json", report)
    return report
