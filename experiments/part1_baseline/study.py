"""Train or reuse both 4,000-state baselines, select on dev, then evaluate the sealed choice."""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import torch

from .evaluate import evaluate
from .model import Network
from .protocol import ERROR_SETS, MODELS, ROOT, digest, put, read

LEGACY_WEIGHTS = "e862b3087a58d8e26be594a016571e51b3712589f54ed454ba72f57592e9c473"


def development(metrics):
    f1 = metrics["verification"]["structured_macro_f1"]
    repair = metrics["candidate_repair"]["success"]
    direct = metrics["pipeline"]["direct"]["success"]
    return dict(score=(direct + f1 + repair) / 3, structured_macro_f1=f1, repair=repair, direct=direct)


def winner(scores):
    return max(
        scores,
        key=lambda name: (
            scores[name]["score"],
            scores[name]["structured_macro_f1"],
            scores[name]["repair"],
            name == "flan_t5_base",
        ),
    )


def export_models(work, models):
    """Install immutable, reloadable checkpoints at the paths used by the prediction CLI."""
    root = work / "models/expanded"
    root.mkdir(parents=True, exist_ok=True)
    for family, (source, weights) in models.items():
        dest = root / family
        if not dest.exists():
            with TemporaryDirectory(dir=root) as tmp:
                stage = Path(tmp) / "model"
                stage.mkdir()
                for item in source.iterdir():
                    if item.is_file():
                        try:
                            os.link(item, stage / item.name)
                        except OSError:
                            shutil.copy2(item, stage / item.name)
                if digest(stage / "model.safetensors") != weights:
                    raise ValueError("Checkpoint changed during export")
                stage.rename(dest)
        if (
            digest(dest / "model.safetensors") != weights
            or digest(dest / "tokenizer.json") != MODELS[family][3]
            or digest(dest / "config.json") != digest(source / "config.json")
        ):
            raise ValueError(f"Invalid exported model at {dest}")


def get_checkpoint(work, bases, family, history, protocol):
    legacy = history is not None and family == "flan_t5_base"
    if legacy:
        root = history / "runs/uniform"
        folder = root / "epoch3/model"
        config = read(root / "settings.json")
        receipt = read(root / "train_done.json")
        weights = receipt["model_sha256"]
        if (
            weights != LEGACY_WEIGHTS
            or config["base_sha256"] != MODELS[family][2]
            or digest(history / "data/train.jsonl") != protocol["training_sources"]["full"]
        ):
            raise ValueError("The archived uniform checkpoint or its original corpus changed")
    else:
        root = work / "runs/expanded" / family
        if not (root / "train_done.json").exists():
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "experiments.part1_baseline.train",
                    "--work",
                    str(work),
                    "--family",
                    family,
                    "--base",
                    str(bases / family),
                ],
                check=True,
            )
        config = read(root / "settings.json")
        receipt = read(root / "train_done.json")
        weights = receipt["weights"]
        folder = root / "epoch3/model"
        if (
            config["rows"] != 36800
            or config["training_sha256"] != protocol["training_sources"]["full"]
            or config["pretrained_sha256"] != MODELS[family][2]
        ):
            raise ValueError("The candidate was trained with a different corpus or base checkpoint")
    if receipt["updates"] != 375 or digest(folder / "model.safetensors") != weights:
        raise ValueError("Invalid third checkpoint")
    return folder, weights


def assess(work, folder, weights, family, splits):
    net = Network(folder)
    net.model.eval()
    result = {split: evaluate(net, work, split, family, weights, full) for split, full in splits}
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return result


def study(work, bases, history=None):
    frozen = read(work / "data/protocol.json")
    if frozen["quota"]["unit"] != "source state" or frozen["quota"]["states"] != 4000:
        raise ValueError("Part 1 uses the instructor-approved 4,000-state protocol")
    models = {f: get_checkpoint(work, bases, f, history, frozen) for f in MODELS}
    export_models(work, models)
    compare_and_test(work, models, frozen)


def compare_and_test(work, models, frozen):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    scores = {}
    for family, (folder, h) in models.items():
        metrics = assess(work, folder, h, family, [("dev", False)])["dev"]
        scores[family] = dict(
            **development(metrics),
            weights=h,
            metrics_sha256=digest(work / "scores" / family / "dev/metrics.json"),
        )
    primary = winner(scores)
    decision = dict(
        protocol=digest(work / "data/protocol.json"),
        dev_inputs=frozen["inputs"]["dev"],
        train_states=4000,
        train_rows=36800,
        rule=frozen["dev_selection"],
        scores=scores,
        primary_model=primary,
    )
    path = work / "selection.json"
    if path.exists() and read(path) != decision:
        raise ValueError("A different formal selection already exists in this directory")
    put(path, decision)
    print(json.dumps(decision), flush=True)
    folder, weights = models[primary]
    assess(work, folder, weights, primary, [(split, True) for split in ERROR_SETS])
    put(
        work / "study_done.json",
        dict(
            selection_sha256=digest(path),
            test_receipts={
                split: digest(work / "scores" / primary / split / "done.json") for split in ERROR_SETS
            },
        ),
    )
    print(json.dumps(dict(done=True, primary_model=primary)), flush=True)


def recheck(work):
    """Cold inference in an empty directory. Compare the actual prediction and score files."""
    started = time.time()
    frozen = read(work / "data/protocol.json")
    models = {}
    for family in MODELS:
        model = work / "models/expanded" / family
        weights = digest(model / "model.safetensors")
        if digest(model / "tokenizer.json") != MODELS[family][3]:
            raise ValueError("Tokenizer differs from the frozen model")
        models[family] = model, weights
    with TemporaryDirectory(prefix="confirmation-", dir=work) as folder:
        fresh = Path(folder)
        (fresh / "data").mkdir()
        for name in ["protocol.json", *(s + ".jsonl" for s in frozen["inputs"])]:
            shutil.copyfile(work / "data" / name, fresh / "data" / name)
        compare_and_test(fresh, models, frozen)
        runs = {}
        same = digest(work / "selection.json") == digest(fresh / "selection.json")
        for done in sorted((fresh / "scores").glob("*/*/done.json")):
            directory = done.parent.relative_to(fresh)
            receipt = read(done)
            checks = {
                f: digest(fresh / directory / f) == digest(work / directory / f) for f in receipt["files"]
            }
            same = same and all(checks.values())
            runs[directory.relative_to("scores").as_posix()] = dict(same_files=checks, **receipt)
        report = dict(
            identical=same,
            gpu=torch.cuda.get_device_name(),
            started_unix=started,
            completed_unix=time.time(),
            selection=digest(fresh / "selection.json"),
            runs=runs,
        )
        if not same:
            shutil.copytree(fresh, work / (fresh.name + "-differences"))
        put(work / "rerun_confirmation.json", report)
        if not same:
            raise ValueError("Independent run differs. Both sets of predictions have been preserved.")
        print(
            json.dumps(dict(identical=True, checked_files=sum(len(r["same_files"]) for r in runs.values()))),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT / "outputs/part1_baseline")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bases", type=Path)
    mode.add_argument(
        "--recheck",
        action="store_true",
        help="Repeat all inference from the exported checkpoints in an empty directory",
    )
    parser.add_argument("--history", type=Path, help="Reuse the original uniform Flan checkpoint")
    args = parser.parse_args()
    if args.recheck:
        recheck(args.work.resolve())
    else:
        study(args.work.resolve(), args.bases.resolve(), args.history.resolve() if args.history else None)
