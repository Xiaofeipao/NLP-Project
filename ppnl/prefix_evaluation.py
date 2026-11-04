"""One audited entry point for full-model and GVR prefix experiments."""
import json
from pathlib import Path

from .core import OfficialOracle
from .data import write_json, write_jsonl
from .experiment_freeze import audit_splits, file_hash, _tree_hashes, ROOT


def prepare(spec):
    if spec.get("family") not in ("full_finetune", "lora_gvr"):
        raise ValueError("family must be full_finetune or lora_gvr")
    cap = spec.get("max_actions", 36)
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError("max_actions must be a positive integer")
    if not isinstance(spec.get("symbolic_only", False), bool):
        raise ValueError("symbolic_only must be boolean")
    if spec.get("symbolic_only") and spec["family"] != "lora_gvr":
        raise ValueError("Symbolic control is available in the GVR comparison")
    config = json.loads(Path(spec["config"]).read_text(encoding="utf-8"))
    config["local_files_only"] = True
    dataset = Path(spec["dataset_root"])
    audit = audit_splits(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    oracle = OfficialOracle(manifest["executor_path"])
    if oracle.sha256 != manifest["executor_sha256"]:
        raise ValueError("Executor differs from dataset manifest")
    traces = {split: file_hash(spec["traces"][split]) for split in ("dev", "test_id", "test_ood")}
    weights = Path(config.get("policy_checkpoint", Path(config["output_root"]) / "flan_t5_small/train/best"))
    model, selection_hash = None, None
    if not spec.get("symbolic_only", False):
        selection = Path(config["output_root"]) / "policy_selection.json"
        selected = json.loads(selection.read_text(encoding="utf-8"))  # Required, never optional.
        if selected.get("weights_sha256") != file_hash(weights / "model.safetensors"):
            raise ValueError("Policy weights differ from development selection")
        selection_hash, model = file_hash(selection), _tree_hashes(weights)
    source = {p.relative_to(ROOT).as_posix(): file_hash(p)
              for folder in ("ppnl", "scripts") for p in sorted((ROOT / folder).rglob("*.py"))}
    identity = {"spec": spec, "config": config, "split_audit": audit,
                "traces_sha256": traces, "model_files": model,
                "selection_sha256": selection_hash, "source_sha256": source,
                "executor_sha256": oracle.sha256}
    return config, oracle, weights, identity


def create_lock(identity, dev_report):
    report = json.loads(Path(dev_report).read_text(encoding="utf-8"))
    if report.get("split") != "dev" or report.get("identity") != identity or not report.get("metrics"):
        raise ValueError("A matching completed dev report is required before freezing")
    return {"schema_version": 1, "identity": identity,
            "dev_report": str(Path(dev_report).resolve()), "dev_sha256": file_hash(dev_report),
            "test_status": "reused_exploratory"}


def validate_lock(path, identity):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("schema_version") != 1 or record.get("identity") != identity:
        raise ValueError("Prefix experiment changed after freezing")
    if record.get("dev_sha256") != file_hash(record["dev_report"]):
        raise ValueError("Development report changed after freezing")
    create_lock(identity, record["dev_report"])
    return {"freeze_sha256": file_hash(path), "test_status": "reused_exploratory"}


def run(spec, split, output, freeze=None):
    if split not in ("dev", "test_id", "test_ood"):
        raise ValueError("Unknown split")
    if split != "dev" and not freeze:
        raise ValueError("Test inference requires --freeze")
    out = Path(output)
    if out.exists():
        raise FileExistsError("Choose a new output directory")
    config, oracle, weights, identity = prepare(spec)
    receipt = validate_lock(freeze, identity) if split != "dev" else None
    from . import full_prefix_repair, prefix_state_repair
    trace_format = spec.get("trace_format", "original" if spec["family"] == "full_finetune" else "gvr")
    if trace_format not in ("original", "gvr"):
        raise ValueError("Unknown trace_format")
    module = full_prefix_repair if trace_format == "original" else prefix_state_repair
    paired = module.load_generated_candidates(spec["dataset_root"], split, spec["traces"][split], oracle)

    def evaluate(scorer, check=lambda: None):
        kwargs = {"methods": ("prefix_visit_heuristic",)} if spec.get("symbolic_only") else {}
        return module.evaluate_generated(paired, scorer, oracle, spec.get("max_actions", 36), check, **kwargs)

    usage = None
    if spec.get("symbolic_only"):
        metrics, records = evaluate(None)
    else:
        from .modeling import Seq2Seq, gpu_budget, require_cuda
        from .policy import NeuralPolicy
        require_cuda()
        with gpu_budget(config["gpu_ledger"], config["gpu_hours_cap"], f"prefix:{spec['family']}:{split}") as check:
            scorer = NeuralPolicy(Seq2Seq(str(weights), config), check)
            metrics, records = evaluate(scorer, check)
            usage = scorer.usage.copy()
    out.mkdir(parents=True, exist_ok=False)
    report = {"split": split, "n": len(paired), "identity": identity, "metrics": metrics,
              "policy_usage": usage, "freeze_receipt": receipt,
              "test_status": "reused_exploratory", "budget_applies_to": "repaired_paths",
              "source": "saved generator traces; state policy is separately trained"}
    write_json(out / "results.json", report)
    write_jsonl(out / "traces.jsonl", records)
    return report
