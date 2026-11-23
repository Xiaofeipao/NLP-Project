"""Verify the shared-state checkpoint, optionally restoring the original single safetensors file."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def verify(model_dir, output=None):
    report = Path(__file__).resolve().parent
    manifest = json.loads((report / "checkpoint_manifest.json").read_bytes())
    source = manifest["source_checkpoint"]
    expected = {f["name"]: f for f in manifest["files"]}
    header = (report / source["header"]).read_bytes()
    digest = hashlib.sha256()
    total = 0
    destination = output.open("xb") if output is not None else None

    def include(data):
        nonlocal total
        digest.update(data)
        total += len(data)
        if destination is not None:
            destination.write(data)

    try:
        include(struct.pack("<Q", len(header)))
        include(header)
        for name in source["shard_payload_order"]:
            file_digest = hashlib.sha256()
            with (model_dir / name).open("rb") as shard:
                prefix = shard.read(8)
                if len(prefix) != 8:
                    raise ValueError(f"Missing safetensors header in {name}")
                length, = struct.unpack("<Q", prefix)
                if length > expected[name]["bytes"] - 8:
                    raise ValueError(f"Invalid safetensors header in {name}. Fetch the Git LFS files first.")
                file_digest.update(prefix)
                file_digest.update(shard.read(length))
                for part in iter(lambda: shard.read(1024 * 1024), b""):
                    file_digest.update(part)
                    include(part)
                if shard.tell() != expected[name]["bytes"] or file_digest.hexdigest() != expected[name]["sha256"]:
                    raise ValueError(f"Shard integrity check failed for {name}")
        if total != source["bytes"] or digest.hexdigest() != source["sha256"]:
            raise ValueError("Reconstructed bytes differ from the training checkpoint")
        if destination is not None:
            destination.close()
    except BaseException:
        if destination is not None:
            destination.close()
            output.unlink(missing_ok=True)
        raise
    print(f"{total} bytes, SHA-256 {digest.hexdigest()}")


if __name__ == "__main__":
    default_dir = Path(__file__).resolve().parent.parent / "models" / "shared4000-epoch3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=default_dir)
    parser.add_argument("--output", type=Path, help="Write the restored checkpoint to a new file.")
    args = parser.parse_args()
    verify(args.model_dir, args.output)
