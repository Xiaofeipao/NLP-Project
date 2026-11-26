"""Compare V few-shot prompts using the published checkpoint and frozen Hope splits."""

import argparse
import json
from collections import Counter
from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace

import torch

from ppnl.core import Grid, execute
from .diagnose import expected_verdict
from .evaluate import evaluate
from .model import Network
from .policy import fields
from .protocol import ERROR_SETS, PROMPTS, ROOT, digest, gold_rows, point, put, read

HERE = Path(__file__).resolve()
PUBLISHED = ROOT / "artifacts/part1_baseline"
VARIANTS = {
    "0-shot": PROMPTS / "V.txt",
    **{f"{n}-shot": HERE.parent / "prompts" / f"v_{n}shot.txt" for n in (2, 3)},
}


def fixed_probe(net, template, split, folder):
    """Probe V on identical correct C coordinates, separately from the model rollouts."""
    gold, texts = [], []
    for row in gold_rows(split):
        grid = Grid.from_dict(row["grid"])
        context = fields(row["grid"])
        for event in execute(grid, row["candidate"]).trace:
            if event["action"] == "<eos>":
                continue
            cell = "no_coordinate" if event["reason"] == "invalid_action_token" else point(event["to"])
            inputs = dict(context, current=cell)
            texts.append(template.format(**inputs))
            gold.append(dict(id=row["id"], step=event["step"], gold=expected_verdict(grid, cell), **inputs))
    predicted = net.predict("V", texts)
    for row, label in zip(gold, predicted, strict=True):
        row["predicted"] = label
    matrix = {}
    for label in sorted({row["gold"] for row in gold}):
        matrix[label] = dict(
            sorted(Counter(row["predicted"] for row in gold if row["gold"] == label).items())
        )
    put(folder / "predictions.jsonl", gold, True)
    return dict(pairs=len(gold), confusion=matrix, predictions_sha256=digest(folder / "predictions.jsonl"))


def study(baseline, out):
    base_protocol_path = baseline / "data/protocol.json"
    base_selection_path = baseline / "selection.json"
    frozen = read(base_protocol_path)
    selection = read(base_selection_path)
    assert selection["protocol"] == digest(base_protocol_path)
    assert all(digest(PROMPTS / f"{task}.txt") == value for task, value in frozen["prompts"].items())
    family = selection["primary_model"]
    checkpoint = baseline / "models/expanded" / family
    weights = digest(checkpoint / "model.safetensors")
    assert family == "flan_t5_base" and weights == selection["scores"][family]["weights"]
    assert digest(checkpoint / "tokenizer.json") == frozen["models"][family][3]
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    network = Network(checkpoint)
    network.model.eval()
    original = VARIANTS["0-shot"].read_text(encoding="utf-8").rstrip("\r\n")
    for variant, prompt in VARIANTS.items():
        full_prompt = prompt.read_text(encoding="utf-8").rstrip("\r\n")
        assert full_prompt.endswith(original)
        prefix = full_prompt[: -len(original)]
        work = out / variant
        if prefix:
            protocol = dict(
                frozen,
                prompts=dict(frozen["prompts"], V=digest(prompt)),
                prompt_variant=variant,
                source_protocol=digest(base_protocol_path),
                source_selection=digest(base_selection_path),
            )
            put(work / "data/protocol.json", protocol)
            put(work / "selection.json", dict(selection, protocol=digest(work / "data/protocol.json")))
            for split in ERROR_SETS:
                copyfile(baseline / "data" / (split + ".jsonl"), work / "data" / (split + ".jsonl"))

            def predict(task, texts):
                return network.predict(task, [prefix + text for text in texts] if task == "V" else texts)

            v_shot = SimpleNamespace(device=network.device, predict=predict)
            for split in ERROR_SETS:
                print(json.dumps(dict(variant=variant, split=split, stage="rollout")), flush=True)
                evaluate(v_shot, work, split, family, weights, full=True)

        for split in ERROR_SETS:
            print(json.dumps(dict(variant=variant, split=split, stage="fixed_probe")), flush=True)
            folder = work / "fixed_probe" / split
            summary = fixed_probe(network, full_prompt, split, folder)
            put(
                folder / "metrics.json",
                dict(
                    summary,
                    prompt=digest(prompt),
                    weights=weights,
                    inputs=frozen["inputs"][split],
                    device=str(network.device),
                ),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=PUBLISHED)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    study(args.baseline.resolve(), args.out.resolve())
