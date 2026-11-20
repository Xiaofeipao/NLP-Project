"""Use these entry points with a StepVerification-v1-v2 Seq2Seq runner."""
import json
from pathlib import Path

from ppnl.data import read_jsonl
from ppnl.experiment_freeze import file_hash
from ppnl.policy import NAMES
from .schema import VERSION, State, render, template_hash


def read_split(dataset, split, *, version=VERSION, fingerprint=None):
    root = Path(dataset)
    if split not in ("train", "dev", "test_id", "test_ood"):
        raise ValueError(split)
    manifest = json.loads((root / "manifest.json").read_bytes())
    if (manifest["schema"], manifest["template_sha256"]) != (version, fingerprint or template_hash()):
        raise ValueError("The renderer and dataset use different prompt versions")
    name = split + ".jsonl"
    if file_hash(root / name) != manifest["sha256"][name]:
        raise ValueError(f"Changed dataset split: {name}")
    return read_jsonl(root / name)


def prepare_training(runner, dataset, *, version=VERSION, fingerprint=None):
    """Return mixed features for the existing optimal-set and split-verdict loss."""
    if runner.policy_loss != "optimal_actions":
        raise ValueError("G requires the full optimal-action set")
    rows = read_split(dataset, "train", version=version, fingerprint=fingerprint)
    features = runner.encode_rows(rows)
    for row, item in zip(rows, features, strict=True):
        expected = [a in row.get("optimal_actions", ()) for a in NAMES]
        if item["policy_actions"] != expected or any(expected) != (row["mode"] == "g_next"):
            raise ValueError(f"Incorrect action-set feature: {row['id']}")
    runner.model.config.ppnl_prompt_sha256 = fingerprint or template_hash()
    return features


def batch_loss(runner, batch):
    """V2's Flan-T5-base loss. No second, constant-target stop loss on G."""
    return runner.loss(batch.to(runner.device), split_verification_loss=True)


def check_checkpoint(runner, fingerprint=None):
    if getattr(runner.model.config, "ppnl_prompt_sha256", None) != (fingerprint or template_hash()):
        raise ValueError("Expected a checkpoint trained with this prompt version")


class StatePolicy:
    """Score the same four moves as v2; usable with ppnl.policy.greedy_plans."""
    def __init__(self, runner, check=lambda: None, *, fingerprint=None):
        check_checkpoint(runner, fingerprint)
        self.runner, self.check = runner, check

    def scores(self, observations):
        runner, results = self.runner, []
        runner.model.eval()
        texts = [render("g_next", State.from_grid(grid, position)) for grid, position in observations]
        size = runner.config["training"]["eval_batch_size"]
        for i in range(0, len(texts), size):
            self.check()
            batch = runner.tokenizer(texts[i:i+size], padding=True, truncation=False, return_tensors="pt")
            if batch.input_ids.shape[1] > runner.input_limit:
                raise ValueError("Policy input exceeds the input limit")
            start = runner.torch.full((len(batch.input_ids), 1), runner.model.config.decoder_start_token_id,
                                      dtype=runner.torch.long, device=runner.device)
            with runner.torch.inference_mode(), runner.autocast():
                scores = runner.model(**batch.to(runner.device), decoder_input_ids=start).logits[:, 0, runner.action_ids]
            results.extend(scores.float().cpu().tolist())
        return results


def verify_plans(runner, grids, plans, check=lambda: None):
    """Signature also fits state_baselines.workflow.compare(..., verifier=verify_plans)."""
    check_checkpoint(runner)
    texts = [render("v_plan", State.from_grid(g), plan) for g, plan in zip(grids, plans, strict=True)]
    return runner.predict(texts, check)


def verify_moves(runner, observations, check=lambda: None):
    check_checkpoint(runner)
    return runner.predict([render("v_step", State.from_grid(g, position), action or "")
                           for g, position, action in observations], check)
