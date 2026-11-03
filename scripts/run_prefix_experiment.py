"""Dev evaluation, freezing and test evaluation for both prefix families."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.prefix_evaluation import prepare, create_lock, run
from ppnl.experiment_freeze import write_new_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "evaluate", "freeze"))
    parser.add_argument("--spec", required=True)
    parser.add_argument("--split", choices=("dev", "test_id", "test_ood"))
    parser.add_argument("--out")
    parser.add_argument("--freeze")
    parser.add_argument("--dev-report")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    if args.command == "check":
        prepare(spec)
        print("Data, traces, executor and development-selected policy verified; no inference")
    elif args.command == "freeze":
        if not args.out or not args.dev_report:
            parser.error("freeze requires --out and --dev-report")
        identity = prepare(spec)[3]
        write_new_json(args.out, create_lock(identity, args.dev_report))
        print("Created prefix experiment freeze")
    else:
        if not args.out or not args.split:
            parser.error("evaluate requires --out and --split")
        result = run(spec, args.split, args.out, args.freeze)
        print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
