"""Run the unified evaluation harness: every planner, every split, one report."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.eval_harness import SPLITS, METHODS, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.policy.best.json")
    parser.add_argument("--split", action="append", choices=list(SPLITS), dest="splits",
                        help="repeatable; defaults to dev, test_id, test_ood")
    parser.add_argument("--methods", default="all",
                        help="comma-separated subset of: " + ", ".join(m.name for m in METHODS))
    parser.add_argument("--out", default=None, help="output directory; defaults to outputs/harness/<timestamp>")
    parser.add_argument("--baseline", default=None,
                        help="experiment root holding frozen eval_<split>/metrics.json; defaults to the config output_root")
    parser.add_argument("--symbolic-only", action="store_true",
                        help="skip model methods; runs on CPU without CUDA")
    parser.add_argument("--allow-new-weights", action="store_true",
                        help="evaluate even when weights differ from the development selection")
    parser.add_argument("--no-direct", action="store_true",
                        help="skip the stored original whole-plan baseline row")
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run(config, splits=tuple(args.splits or SPLITS), methods=args.methods, out=args.out,
                 baseline=args.baseline, allow_new_weights=args.allow_new_weights,
                 symbolic_only=args.symbolic_only, with_direct=not args.no_direct,
                 write_html=not args.no_html, config_path=args.config)
    print(json.dumps({"output_dir": result["output_dir"],
                      "comparison": {split: entry["comparison"]["status"]
                                     for split, entry in result["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
