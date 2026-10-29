"""One frozen generation base and independently trained verifier/repair LoRA weights."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

from .modeling import Seq2Seq
from .prompts import prompt


def checkpoint_hash(path):
    root = Path(path)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".bin", ".safetensors"))
    if not files:
        raise ValueError(f"No checkpoint weights in {root}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(root).as_posix().encode())
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def lora_config(settings):
    from peft import LoraConfig, TaskType
    return LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, target_modules=["q", "v"],
                      r=settings.get("r", 8), lora_alpha=settings.get("alpha", 16),
                      lora_dropout=settings.get("dropout", 0.05), bias="none")


def attach_adapter(runner, name, settings):
    from peft import get_peft_model
    if name not in ("verification", "repair"):
        raise ValueError(name)
    targets = [n for n, _ in runner.model.named_modules() if n.endswith((".q", ".v"))]
    if not targets:
        raise ValueError("No T5 attention q/v modules found")
    runner.model = get_peft_model(runner.model, lora_config(settings), adapter_name=name)
    runner._collator.model = runner.model
    trainable = [n for n, p in runner.model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n or name not in n for n in trainable):
        raise ValueError("Unexpected trainable non-adapter parameters")
    return {"matched_modules": len(targets), "trainable_parameters": sum(p.numel() for p in runner.model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in runner.model.parameters())}


class GVRModel:
    """Named adapter switching. The generator always runs with adapters disabled."""
    def __init__(self, base, adapters, config):
        from peft import PeftModel
        self.base_hash = checkpoint_hash(base)
        self.runner = Seq2Seq(str(base), config)
        self.adapter_hashes = {}
        if adapters:
            for task, directory in adapters.items():
                if task not in ("verification", "repair"):
                    raise ValueError(task)
                meta = json.loads((Path(directory) / "base_identity.json").read_text(encoding="utf-8"))
                if meta["stage"] != task or meta["generation_base_sha256"] != self.base_hash:
                    raise ValueError("Adapter was trained on a different generation base")
                self.adapter_hashes[task] = checkpoint_hash(directory)
                if task == next(iter(adapters)):
                    self.runner.model = PeftModel.from_pretrained(self.runner.model, directory,
                                                                   adapter_name=task, is_trainable=False)
                else:
                    self.runner.model.load_adapter(directory, adapter_name=task, is_trainable=False)
            self.runner._collator.model = self.runner.model

    def predict(self, task, grid, candidate="", feedback="", check=lambda: None):
        if task not in ("generation", "verification", "repair"):
            raise ValueError(task)
        if task != "generation" and task not in self.adapter_hashes:
            raise ValueError(f"Missing {task} adapter")
        model = self.runner.model
        if task == "generation" and self.adapter_hashes:
            context = model.disable_adapter()
        else:
            if task != "generation":
                model.set_adapter(task)
            context = nullcontext()
        with context:
            source = prompt(task, grid, candidate, feedback)
            result = self.runner.predict([source], check)[0]
        self.last_usage = {"input_tokens": len(self.runner.tokenizer(source)["input_ids"]),
                           "output_tokens": len(self.runner.tokenizer(result)["input_ids"])}
        return result
