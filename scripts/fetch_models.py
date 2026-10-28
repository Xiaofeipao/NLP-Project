"""Download the two exact configured checkpoints; no CUDA or training."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download
    results = {}
    for key, model in config["models"].items():
        print(f"Downloading {model['checkpoint']}", flush=True)
        path = snapshot_download(model["checkpoint"], revision=model.get("revision"),
                                 cache_dir=config.get("model_cache", ".cache/huggingface"),
                                 max_workers=2,
                                 allow_patterns=["config.json", "generation_config.json", "tokenizer*",
                                                 "special_tokens_map.json", "spiece.model", "vocab.json",
                                                 "merges.txt", "model.safetensors", "added_tokens.json"])
        results[key] = {"checkpoint": model["checkpoint"], "snapshot": str(Path(path).resolve()),
                        "revision": Path(path).name}
        print(results[key], flush=True)
    target = Path(config.get("model_cache", ".cache/huggingface")) / "download_manifest.json"
    target.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
