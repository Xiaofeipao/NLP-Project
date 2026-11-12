"""Legacy freeze for the GVR repair study schema at commit 23183e2.

Requires the archived development drivers and their historical output layout.
New studies use run_post_training.py and its own protocol.json; their reports
must not be supplied to this legacy freeze procedure.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppnl.experiment_freeze import write_new_json
from ppnl.gvr_study_lock import development_report, study_identity
from ppnl.prefix_evaluation import create_lock

ROOT = Path(__file__).resolve().parents[1]


def main():
    print("Legacy GVR freeze (23183e2 output schema); new studies use run_post_training.py.",
          file=sys.stderr)
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
