"""Audit splits or freeze/check a selected GVR experiment without GPU inference."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.experiment_freeze import audit_splits, create_freeze, validate_freeze, write_new_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "freeze", "check"))
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--protocol", default="config.gvr.next.json")
    parser.add_argument("--out", required=True, help="New output, or existing lock for check")
    parser.add_argument("--selection-report", help="Successful verify_dev.json for selected weights")
    parser.add_argument("--note", default="", help="Explain development-only selection")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    if args.command == "check":
        print(json.dumps(validate_freeze(args.out, config, protocol), indent=2))
        return
    if args.command == "audit":
        result = audit_splits(protocol["dataset_root"])
    else:
        if not args.selection_report or not args.note.strip():
            parser.error("freeze requires --selection-report and --note")
        result = create_freeze(config, protocol, args.selection_report, args.note)
    write_new_json(args.out, result)
    print(f"Created {args.out}")


if __name__ == "__main__":
    main()
