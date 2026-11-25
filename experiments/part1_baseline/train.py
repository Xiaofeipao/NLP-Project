"""Full-parameter uniform G/C/V baseline. Train only on the registered prompt/target file."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch

from .model import Network
from .protocol import LABELS, MODELS, ROOT, SEED, UPDATES, digest, put, read, rows_at, schedule


def train(work, family, base):
    freeze = read(work / "data/protocol.json")
    path = ROOT / "data/coordinate_gcv_review/train.jsonl"
    source = freeze["training_sources"]["full"]
    _, revision, base_hash, tok_hash = MODELS[family]
    if (
        digest(path) != source
        or digest(base / "model.safetensors") != base_hash
        or digest(base / "tokenizer.json") != tok_hash
        or (base / "upstream_revision.txt").read_text().strip() != revision
    ):
        raise ValueError("Training inputs or pretrained weights changed")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The registered training recipe needs a BF16 CUDA GPU")
    out = work / "runs/expanded" / family
    out.mkdir(parents=True, exist_ok=True)
    rows = rows_at(path)
    assert len(rows) == 36800
    ids = work / "data/state_ids.json"
    if (
        digest(ids) != freeze["states_sha256"]
        or freeze["quota"]["unit"] != "source state"
        or len(set(r["case"] for r in rows)) != 4000
        or set(read(ids)) != {r["case"] for r in rows}
    ):
        raise ValueError("Training rows do not match the registered 4,000 source states")
    batches, weights, normalizer = schedule(rows)
    batch_hash = hashlib.sha256(
        json.dumps([b for _, _, b in batches], separators=(",", ":")).encode()
    ).hexdigest()
    seen = Counter(i for _, _, batch in batches for i in batch)
    assert all(seen[i] == (1 if r["task"] == "C" else 2) for i, r in enumerate(rows))

    epochs = [out / f"epoch{i}" for i in range(1, 4)]
    n = sum(d.exists() for d in epochs)
    if [d.exists() for d in epochs] != [i < n for i in range(3)]:
        raise ValueError("Checkpoint gap")
    for i, d in enumerate(epochs[:n], 1):
        receipt = read(d / "done.json")
        if receipt["updates"] != i * UPDATES or digest(d / "model/model.safetensors") != receipt["weights"]:
            raise ValueError("Invalid resumable checkpoint")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(SEED)
    net = Network(epochs[n - 1] / "model" if n else base)
    model = net.model
    assert all(p.requires_grad for p in model.parameters())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    encoded = net.encode([r["text"] for r in rows])
    settings = dict(
        family=family,
        tier="expanded",
        seed=SEED,
        lr=1e-4,
        weight_decay=0.01,
        clip=1.0,
        microbatch=4,
        denominator=normalizer,
        updates=len(batches),
        pretrained_sha256=base_hash,
        revision=revision,
        tokenizer_sha256=tok_hash,
        training_sha256=source,
        order_sha256=batch_hash,
        parameters=sum(p.numel() for p in model.parameters()),
        rows=len(rows),
        presentations=dict(Counter(rows[i]["task"] for _, _, b in batches for i in b)),
        V_first_ids=dict(zip(LABELS, net.label_ids, strict=True)),
        decoder_prefix=list(net.prefix),
        longest_input=max(map(len, encoded)),
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
    )
    if (out / "settings.json").exists() and read(out / "settings.json") != settings:
        raise ValueError("Resume settings differ")
    put(out / "settings.json", settings)
    opts, classes = [], []
    for row in rows:
        cls = LABELS.index(row["answers"][0]) if row["task"] == "V" else None
        classes.append(cls)
        opts.append([(net.label_ids[cls],)] if cls is not None else [net.target(s) for s in row["answers"]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, fused=True)
    if 0 < n < 3:
        optimizer.load_state_dict(
            torch.load(epochs[n - 1] / "optimizer.pt", map_location="cpu", weights_only=True)
        )
    w = torch.tensor(weights, device="cuda", dtype=torch.float32)
    model.train()
    watch = next(
        p
        for name, p in model.named_parameters()
        if "q_proj.weight" in name or "SelfAttention.q.weight" in name
    )
    for i in range(n, 3):
        start = time.perf_counter()
        for k in range(i * UPDATES, (i + 1) * UPDATES):
            stream, idx, batch = batches[k]
            torch.manual_seed(SEED + idx + (10000 if stream == "GV" else 0))
            optimizer.zero_grad(set_to_none=True)
            initial = watch[:8, :16].detach().clone() if idx == 0 else None
            total = torch.zeros((), device="cuda")
            for offset in range(0, len(batch), 4):
                group = batch[offset : offset + 4]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    terms = net.loss(
                        [encoded[j] for j in group], [opts[j] for j in group], [classes[j] for j in group]
                    )
                    loss = terms.dot(w[group]) / normalizer
                loss.backward()
                total += loss.detach()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            delta = (watch[:8, :16].detach() - initial).abs().max().item() if initial is not None else None
            if delta is not None and delta <= 0:
                raise RuntimeError("No parameter change at first update")
            if k < 3 or (k + 1) % 10 == 0 or (k + 1) % UPDATES == 0:
                progress = dict(
                    epoch=i + 1,
                    update=k + 1,
                    stream=stream,
                    loss=total.item(),
                    grad_norm=norm.item(),
                    parameter_change=delta,
                    epoch_elapsed_s=round(time.perf_counter() - start, 2),
                    peak_cuda_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                )
                put(out / "progress.json", progress)
                print(json.dumps(dict(family=family, **progress)), flush=True)
        optimizer.zero_grad(set_to_none=True)
        with TemporaryDirectory(dir=out) as tmp:
            stage = Path(tmp) / "checkpoint"
            stage.mkdir()
            net.save(base, stage / "model")
            if i < 2:
                torch.save(optimizer.state_dict(), stage / "optimizer.pt")
            receipt = dict(
                epoch=i + 1,
                updates=(i + 1) * UPDATES,
                weights=digest(stage / "model/model.safetensors"),
                gpu_wall_seconds=time.perf_counter() - start,
            )
            put(stage / "done.json", receipt)
            stage.rename(epochs[i])
    put(out / "train_done.json", read(epochs[-1] / "done.json"))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--work", type=Path, default=ROOT / "outputs/part1_baseline")
    cli.add_argument("--family", choices=MODELS, required=True)
    cli.add_argument("--base", type=Path, required=True)
    args = cli.parse_args()
    train(args.work.resolve(), args.family, args.base.resolve())
