"""Publish small RLVR evidence only; weights and private outputs stay local."""
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.experiment_freeze import file_hash, write_new_json


def main():
    source = Path("outputs/rlvr_reinforce_001")
    target = Path("artifacts/rlvr")
    if target.exists() and set(p.name for p in target.iterdir()) != {"README.md"}:
        raise FileExistsError(target)
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    normalized = {}
    for name, digest in provenance["source"].items():
        portable = name.replace("\\", "/")
        portable = portable[portable.index("scripts/"):] if "scripts/" in portable else portable
        if file_hash(portable) != digest:
            raise ValueError(f"Recorded training source changed: {portable}")
        normalized[portable] = digest
    provenance["source"] = normalized
    provenance["base_source_commit_at_run"] = "8e05270098ea642473e9b617ace3f29d43cf79d1"
    provenance["publication_note"] = "Paths normalized; source snapshot covers RL training additions. Existing dependencies at base_source_commit_at_run. No weights exported."
    target.mkdir(parents=True, exist_ok=True)
    for name in ("dev_0.json", "dev_16.json", "dev_32.json", "dev_48.json", "dev_64.json",
                 "report.json", "rollouts.jsonl", "reload_check.json"):
        shutil.copyfile(source / name, target / name)
    write_new_json(target / "provenance.json", provenance)
    for name in normalized:
        snapshot = target / "source" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, snapshot)
    write_new_json(target / "manifest.json", {
        "schema_version": 1, "weights_included": False,
        "files": {p.relative_to(target).as_posix(): file_hash(p)
                  for p in sorted(target.rglob("*")) if p.is_file() and p.name != "README.md"}})
    print(f"Exported non-weight evidence to {target}")


if __name__ == "__main__":
    main()
