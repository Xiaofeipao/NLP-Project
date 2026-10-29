"""Opt-in GVR objectives and development-only checkpoint ranking."""
from __future__ import annotations

import math


def stage_options(protocol, stage):
    known = {"generation", "verification", "repair"}
    if stage not in known or protocol.get("stages", {}).keys() - known:
        raise ValueError("Unknown GVR stage")
    options = {"loss": "token_mean", "status_weight": 0.0, "selection": "dev_loss"}
    overrides = protocol.get("stages", {}).get(stage, {})
    unknown = overrides.keys() - {"loss", "status_weight", "selection", "learning_rate"}
    if unknown:
        raise ValueError(f"Unknown GVR stage settings: {sorted(unknown)}")
    options.update(overrides)
    if options["loss"] not in ("token_mean", "sequence_and_status"):
        raise ValueError("Unknown GVR loss")
    if options["selection"] not in ("dev_loss", "task_metrics"):
        raise ValueError("Unknown GVR checkpoint selection")
    weight = options["status_weight"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError("status_weight must be finite and nonnegative")
    if options["loss"] == "sequence_and_status" and (stage != "verification" or weight <= 0):
        raise ValueError("Status supervision requires verification and a positive weight")
    if options["loss"] == "token_mean" and weight != 0:
        raise ValueError("status_weight requires sequence_and_status")
    if "learning_rate" in options:
        rate = options["learning_rate"]
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Stage learning rate must be finite and positive")
    return options


def sequence_status_loss(logits, labels, status_weight):
    """Equal weight per example, plus supervision of the first verdict token.

    The caller checks that valid/invalid verdicts have distinct first tokens.
    Remaining tokens still supervise the failure step, reason and EOS.
    """
    import torch.nn.functional as F

    if labels.ndim != 2 or labels.shape[1] == 0 or (labels[:, 0] == -100).any():
        raise ValueError("Verification labels need an unmasked first token")
    mask = labels != -100
    losses = F.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
    sequence_mean = (losses.sum(dim=1) / mask.sum(dim=1)).mean()
    return sequence_mean + status_weight * losses[:, 0].mean()


def training_loss(runner, batch, options):
    if options["loss"] == "token_mean":
        return runner.loss(batch)
    output = runner.model(**batch)
    return sequence_status_loss(output.logits, batch["labels"], options["status_weight"])


def checkpoint_rank(stage, dev_loss, diagnostics, options):
    """Higher is better; only development data may be passed here."""
    if not math.isfinite(dev_loss):
        raise ValueError("Non-finite development loss")
    if options["selection"] == "dev_loss":
        return (-dev_loss,)
    fields = (("macro_f1", "first_error_localization_accuracy", "failure_type_accuracy")
              if stage == "verification" else ("success_rate", "optimality"))
    values = tuple(diagnostics[field] for field in fields)
    if any(value is None or not math.isfinite(value) for value in values):
        raise ValueError("Missing or non-finite development selection metric")
    return values + (-dev_loss,)
