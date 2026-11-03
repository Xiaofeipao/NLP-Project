"""Adapter from the original full fine-tuned Seq2Seq checkpoint to G/V/R sessions."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .harness import fingerprint
from .prompts import prompt


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(checkpoint, config):
    """Bind a session to one local full checkpoint, config, code and runtime."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir() or (checkpoint / "adapter_config.json").exists():
        raise ValueError("Expected a local full fine-tuned checkpoint, not a LoRA adapter")
    files = sorted(p for p in checkpoint.rglob("*") if p.is_file())
    if not (checkpoint / "config.json").is_file() or not any(
            p.name in ("model.safetensors", "pytorch_model.bin") or
            p.name.startswith(("model-", "pytorch_model-")) and
            p.name.endswith((".safetensors", ".bin")) for p in files):
        raise ValueError("Checkpoint needs config.json and full model weights")
    source = Path(__file__).parent
    code = ("full_finetune_gvr.py", "gvr_harness.py", "harness.py", "core.py",
            "prompts.py", "modeling.py")
    packages = {}
    for name in ("torch", "transformers", "tokenizers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"kind": "full_finetune_gvr", "config_sha256": fingerprint(config),
            "checkpoint": {p.relative_to(checkpoint).as_posix(): _sha256(p) for p in files},
            "code": {name: _sha256(source / name) for name in code}, "packages": packages}


class FullFinetuneGVRModel:
    """Reuse the original multi-task model and exact training/evaluation prompts."""

    def __init__(self, runner):
        self.runner = runner
        self.last_usage = {}

    def predict(self, task, grid, candidate="", feedback="", check=lambda: None):
        source = prompt(task, grid, candidate, feedback)
        result = self.runner.predict([source], check)[0]
        tokenizer = self.runner.tokenizer
        self.last_usage = {"input_tokens": len(tokenizer(source)["input_ids"]),
                           "output_tokens": len(tokenizer(result)["input_ids"])}
        return result
