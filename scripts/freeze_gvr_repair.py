"""Freeze the GVR repair study after all development gates pass."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppnl.experiment_freeze import write_new_json
from ppnl.gvr_study_lock import development_report, study_identity
from ppnl.prefix_evaluation import create_lock

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "outputs/gvr_repair_study"
    report = out / "dev_decision.json"
    lock = out / "freeze.json"
    if report.exists() or lock.exists():
        raise FileExistsError("Development decision or freeze already exists")
    identity = study_identity(ROOT)
    write_new_json(report, development_report(ROOT, identity))
    write_new_json(lock, create_lock(identity, report))
    print(lock)


if __name__ == "__main__":
    main()
