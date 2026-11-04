"""CPU-only preview of paired input formats; never opens test examples."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.data import read_jsonl
from ppnl.prompts import format_training_rows
from ppnl.experiment_freeze import write_new_json, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/derived")
    parser.add_argument("--tokenizer", help="Optional local tokenizer directory; no model loading")
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    report = {"training_run": False, "test_examples_opened": False, "splits": {}}
    for split in ("train", "dev"):
        path = Path(args.dataset) / f"{split}.jsonl"
        rows = read_jsonl(path)
        variants = {style: format_training_rows(rows, style) for style in ("vanilla", "structured")}
        result = {"source_sha256": file_hash(path), "counts": dict(Counter(r["task"] for r in rows)),
                  "examples": {task: {style: next(r["input"] for r in records if r["task"] == task)
                                      for style, records in variants.items()}
                               for task in ("generation", "verification", "repair")}}
        if tokenizer:
            result["max_input_tokens"] = {}
            for style, records in variants.items():
                lengths = [len(ids) for ids in tokenizer([r["input"] for r in records], truncation=False)["input_ids"]]
                result["max_input_tokens"][style] = max(lengths)
                if max(lengths) > args.max_input_tokens:
                    raise ValueError(f"{style}/{split} exceeds token budget; truncation forbidden")
        report["splits"][split] = result
    write_new_json(args.out, report)
    print(json.dumps({s: {k: v for k, v in r.items() if k != "examples"}
                      for s, r in report["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
