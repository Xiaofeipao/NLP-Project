"""Reload selected RL adapter and reproduce saved dev predictions on CUDA."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.data import read_jsonl, write_json
from ppnl.core import Grid
from ppnl.prompts import prompt
from ppnl.modeling import Seq2Seq, gpu_budget, require_cuda
from ppnl.task_adapters import checkpoint_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="configs/rlvr.json")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    config = json.loads(Path(spec["config"]).read_text())
    out = Path(spec["output"])
    if (out / "reload_check.json").exists():
        raise FileExistsError("Reload check already exists")
    report = json.loads((out / "report.json").read_text())
    expected = json.loads((out / f"dev_{report['selected']['step']}.json").read_text())
    rows = [r for r in read_jsonl(Path(config["dataset_root"]) / "dev.jsonl") if r["task"] == "generation"]
    if [r["id"] for r in rows] != [c["id"] for c in expected["cases"]]:
        raise ValueError("Dev membership changed")
    require_cuda()
    import torch
    torch.set_num_threads(1)
    from peft import PeftModel
    with gpu_budget(config["gpu_ledger"], config["gpu_hours_cap"], "rlvr:reload_check:dev_only") as check:
        runner = Seq2Seq(spec["base"], config)
        runner.output_limit = spec["max_new_tokens"]
        runner.model = PeftModel.from_pretrained(runner.model, out / "best")
        plans = runner.predict([prompt("generation", Grid.from_dict(r["grid"])) for r in rows], check)
        matches = plans == [c["plan"] for c in expected["cases"]]
        write_json(out / "reload_check.json", {"matches": matches, "n": len(plans),
                   "adapter_sha256": checkpoint_hash(out / "best"), "base_sha256": checkpoint_hash(spec["base"])})
        if not matches:
            raise RuntimeError("Reload predictions differ")
        print("Selected adapter reload reproduced all dev predictions")


if __name__ == "__main__":
    main()
