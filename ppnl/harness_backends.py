"""Adapters for existing trained models; these models do not natively call tools."""
from pathlib import Path
import hashlib
import json
from importlib.metadata import version

from .core import ACTIONS, Grid, execute
from .harness import fingerprint
from .prompts import prompt


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def backend_identity(spec, config):
    checkpoint = Path(spec["checkpoint"])
    files = sorted(p for p in checkpoint.rglob("*") if p.is_file())
    if not files or not any(p.name.endswith((".safetensors", ".bin")) for p in files):
        raise ValueError("Expected an existing local checkpoint, not a download identifier")
    modules = ("harness.py", "harness_backends.py", "core.py", "prompts.py", "policy.py", "modeling.py")
    recovery = None
    if spec["backend"] == "repair_policy":
        recovery_config = json.loads(Path(spec["recovery_config"]).read_text(encoding="utf-8"))
        recovery_config["local_files_only"] = True
        recovery_spec = {"backend": "policy", "checkpoint": spec["recovery_checkpoint"],
                         "legal_mask": True, "avoid_revisits": spec.get("avoid_revisits", False)}
        recovery = backend_identity(recovery_spec, recovery_config)
    return {"kind": spec["backend"], "spec_sha256": fingerprint(spec), "recovery": recovery,
            "model_config_sha256": fingerprint(config),
            "packages": {name: version(name) for name in ("torch", "transformers", "tokenizers")},
            "files": {p.relative_to(checkpoint).as_posix(): file_hash(p) for p in files},
            "code": {name: file_hash(Path(__file__).with_name(name)) for name in modules}}


class PolicyBackend:
    """One scored action per turn; optional legal mask, no graph search."""

    def __init__(self, scorer, legal_mask=False, avoid_revisits=False):
        if avoid_revisits and not legal_mask:
            raise ValueError("avoid_revisits requires legal_mask")
        self.scorer, self.legal_mask, self.avoid_revisits = scorer, legal_mask, avoid_revisits

    def next_call(self, context):
        grid = Grid.from_dict(context["grid"])
        position = tuple(context["position"])
        scores = self.scorer.scores([(grid, position)])[0]
        choices = list(ACTIONS)
        if self.legal_mask:
            from .policy import neighbors
            choices = list(neighbors(grid, position))
            if choices and self.avoid_revisits:
                from collections import Counter
                visits = Counter([grid.start])
                for event in execute(grid, context["state"]["plan"]).trace:
                    if event["action"] in ACTIONS and event["reason"] is None:
                        visits[tuple(event["to"])] += 1
                legal = neighbors(grid, position)
                minimum = min(visits[legal[a]] for a in choices)
                choices = [a for a in choices if visits[legal[a]] == minimum]
        if not choices:
            return {"name": "finish", "arguments": {}}
        action = max(choices, key=lambda a: scores[list(ACTIONS).index(a)])
        return {"name": "move", "arguments": {"action": action}}


class RepairBackend:
    """Whole-plan generation followed by executor-feedback repair prompts."""

    def __init__(self, runner, check):
        self.runner, self.check = runner, check

    def next_call(self, context):
        state = context["state"]
        grid = Grid.from_dict(context["grid"])
        task = "repair" if state["tool_calls"] else "generation"
        text = prompt(task, grid, state["candidate"], state["feedback"])
        plan = self.runner.predict([text], self.check)[0]
        return {"name": "submit_plan", "arguments": {"plan": plan}}


class RepairPolicyBackend:
    """Explicit learned recovery, not improvement of the original model weights.

    All phase decisions derive from persisted state so reconstruction after a
    crash cannot reset the repair allowance. Each next_call invokes one model.
    """

    def __init__(self, repair, policy, repair_calls=2):
        if isinstance(repair_calls, bool) or not isinstance(repair_calls, int) or repair_calls < 1:
            raise ValueError("repair_calls must be a positive integer")
        self.repair, self.policy, self.repair_calls = repair, policy, repair_calls

    def next_call(self, context):
        state = context["state"]
        if state["tool_calls"] < self.repair_calls and not state["plan"]:
            return self.repair.next_call(context)
        return self.policy.next_call(context)


def load_backend(spec, config, check):
    # Only called by explicit run/resume commands, never by inspection.
    from .modeling import Seq2Seq, require_cuda
    require_cuda()
    runner = Seq2Seq(spec["checkpoint"], config)
    if spec["backend"] in ("policy", "repair_policy"):
        from .policy import NeuralPolicy
        if spec["backend"] == "policy":
            return PolicyBackend(NeuralPolicy(runner, check), spec.get("legal_mask", False),
                                 spec.get("avoid_revisits", False))
        recovery_config = json.loads(Path(spec["recovery_config"]).read_text(encoding="utf-8"))
        recovery_config["local_files_only"] = True
        recovery_runner = Seq2Seq(spec["recovery_checkpoint"], recovery_config)
        policy = PolicyBackend(NeuralPolicy(recovery_runner, check), True, spec.get("avoid_revisits", False))
        return RepairPolicyBackend(RepairBackend(runner, check), policy, spec.get("repair_calls", 2))
    if spec["backend"] == "repair":
        return RepairBackend(runner, check)
    raise ValueError("Unknown backend")
