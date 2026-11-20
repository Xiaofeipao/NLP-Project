"""Check frozen V predictions and the production G/V loaders without updating weights."""
import argparse
import gc
from pathlib import Path

import torch
from ppnl.data import read_jsonl, write_json
from ppnl.experiment_freeze import file_hash
from state_baselines.run import CONFIG
from step_data_v3 import g_probes
from step_data_v3.five_class import CLASS_LABELS, FiveClassRunner
from step_data_v3.model_io import check_checkpoint
from step_data_v3.train_gpu import TextVerifierRunner, split, validate_inputs


def check_runner(runner, checkpoint, dataset, states, train, dev):
    epoch = int(checkpoint.name.rsplit("-", 1)[1])
    saved = read_jsonl(checkpoint.parent / f"train_dev_epoch_{epoch}.jsonl")
    check_checkpoint(runner, g_probes.template_hash())
    assert runner.model.config.ppnl_verifier_loss == runner.verifier_loss
    features = runner.prepare_training(dataset)
    assert len(features) == len(train)
    indices = [next(i for i, r in enumerate(train) if len(r.get("optimal_actions", [])) > 1)]
    for label in CLASS_LABELS:
        position = next(i for i, r in enumerate(train) if r["mode"] == "v_step" and r["target"] == label)
        indices.append(position)
    batch = runner.collate([features[i] for i in indices])
    runner.model.eval()
    with torch.inference_mode(), runner.autocast():
        public = g_probes.batch_loss(runner, batch)
        direct = runner.batch_loss(batch)
    assert torch.isfinite(direct) and direct.device.type == "cuda"
    torch.testing.assert_close(public, direct)
    converted = [(states[r["state_id"]], r["candidate"]) for r in dev]
    assert all(g_probes.step_input(s, a) == r["input"] for (s, a), r in zip(converted, dev, strict=True))
    decisions = g_probes.verify_moves(runner, converted)
    actual = [(r["id"], r["target"], p) for r, p in zip(dev, decisions, strict=True)]
    expected = [(r["id"], r["target"], r["prediction"]) for r in saved]
    assert actual == expected
    return {
        "checkpoint": str(checkpoint), "verifier_loss": runner.verifier_loss,
        "train_features": len(features), "mixed_batch_row_ids": [train[i]["id"] for i in indices],
        "finite_cuda_mixed_loss": direct.item(), "public_and_direct_loss_equal": True,
        "prior_dev_decisions_reproduced": len(expected), "dev_accuracy_count": sum(y == p for _, y, p in actual),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "pretrained", "text", "head", "out"):
        cli.add_argument("--" + key, required=True, type=Path)
    args = cli.parse_args()
    states = validate_inputs(args.data, args.pretrained)
    train = split(args.data, "train")
    dev = [r for r in split(args.data, "dev") if r["mode"] == "v_step"]
    precision = "bfloat16" if torch.cuda.is_bf16_supported() else "float32"
    training = dict(CONFIG["training"], precision=precision, eval_batch_size=16,
                    max_input_tokens=128, max_target_tokens=8)
    report = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "precision": precision,
              "dataset_manifest_sha256": file_hash(args.data / "manifest.json"), "optimizer_updates": 0}
    for key, cls, checkpoint in (("text", TextVerifierRunner, args.text), ("five_class", FiveClassRunner, args.head)):
        runner = cls(str(checkpoint), dict(CONFIG, training=training, local_files_only=True))
        report[key] = check_runner(runner, checkpoint, args.data, states, train, dev)
        del runner
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.out, report)


if __name__ == "__main__":
    main()
