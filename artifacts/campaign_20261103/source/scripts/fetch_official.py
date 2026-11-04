"""Download only pinned SG files and executor. Run manually when ready."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = "5fd0376fc50ae9860bc986085165ebac5675fd44"
BASE = f"https://raw.githubusercontent.com/MohamedAghzal/llms-as-path-planners/{REVISION}/ppnl-spatial-temporal-reasoning"
FILES = {
    "evaluate/executor-point-sg.py": "vendor/ppnl/executor-point-sg.py",
    "single_goal/1_train_set_6x6_samples.json": "data/official/1_train_set_6x6_samples.json",
    "single_goal/1dev_set_6x6_samples.json": "data/official/1dev_set_6x6_samples.json",
    "single_goal/1_goals_test_seen_6x6_samples.json": "data/official/1_goals_test_seen_6x6_samples.json",
    "single_goal/1goals_unseen_6x6_samples.json": "data/official/1goals_unseen_6x6_samples.json",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest_path = root / "vendor" / "ppnl" / "download_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    known = {f["path"]: f for f in previous.get("files", [])} if previous.get("revision") == REVISION else {}
    manifest = {"revision": REVISION, "files": []}
    for source, destination in FILES.items():
        url = f"{BASE}/{source}"
        target = root / destination
        existing = target.read_bytes() if target.exists() else None
        recorded = known.get(destination)
        if existing is not None and recorded and hashlib.sha256(existing).hexdigest() == recorded["sha256"]:
            manifest["files"].append(recorded)
            print(f"Verified existing: {destination}", flush=True)
            continue
        with urlopen(url, timeout=120) as response:
            content = response.read()
        if source.endswith(".json"):
            value = json.loads(content)
            if not isinstance(value, list) or not value or "world" not in value[0]:
                raise ValueError(f"Unexpected upstream data schema: {source}")
        if existing is not None and existing != content:
            raise ValueError(f"Existing file differs from pinned upstream; refusing overwrite: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if existing is None:
            target.write_bytes(content)
        manifest["files"].append({"url": url, "path": destination,
                                   "sha256": hashlib.sha256(content).hexdigest()})
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(destination)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
