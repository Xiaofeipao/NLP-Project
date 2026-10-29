"""Bounded, journaled agent runtime. No model imports or arbitrary code tools.

Backends propose calls; the harness alone owns environment state and grading.
Model calls are at-least-once on recovery; committed moves are replayed from disk.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

from .core import ACTIONS, Grid, execute

VERSION = 1
TOOLS = ("move", "verify_plan", "submit_plan", "finish")


def fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Budgets:
    model_calls: int = 36
    tool_calls: int = 36
    action_attempts: int = 128
    active_seconds: float = 300.0
    context_events: int = 8
    plan_tokens: int = 128

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid budget: {key}")
            if key != "active_seconds" and not isinstance(value, int):
                raise ValueError(f"Budget must be an integer: {key}")


@contextmanager
def _lock(directory):
    # A retained lock file is harmless: the OS releases the actual lock on exit.
    with (Path(directory) / ".lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    # Windows readers and scanners can briefly deny replacement of a snapshot.
    # Retry only sharing/permission failures; keep the durable journal and temp
    # file intact if access remains denied.
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.01 * 2 ** attempt)


def create_session(directory, grid, budgets, backend_identity, oracle_identity):
    """Create an isolated trial. Never reuse or overwrite an existing directory."""
    manifest = {"version": VERSION, "grid": asdict(grid), "budgets": asdict(budgets),
                "backend": backend_identity, "oracle": oracle_identity}
    manifest["fingerprint"] = fingerprint(manifest)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    _atomic_json(directory / "manifest.json", manifest)
    return manifest


def read_session(directory):
    """Read without a model. Journal is authoritative; snapshot is only a view.

    A torn final line fails explicitly instead of silently forgetting a move.
    Inspection during a concurrent append may need to be retried.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    content = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if manifest.get("version") != VERSION or fingerprint(content) != manifest["fingerprint"]:
        raise ValueError("Unsupported or modified session manifest")
    state = {"status": "ready", "plan": "", "model_calls": 0, "tool_calls": 0,
             "action_attempts": 0, "active_seconds": 0.0, "invalid_calls": 0,
             "feedback": "", "candidate": "", "history": []}
    previous = manifest["fingerprint"]
    count = 0
    journal = directory / "events.jsonl"
    if journal.exists():
        with journal.open(encoding="utf-8") as handle:
            for count, line in enumerate(handle, 1):
                if not line.endswith("\n"):
                    raise ValueError("Incomplete journal tail; preserve it before manual recovery")
                event = json.loads(line)
                digest = event.pop("hash")
                if event["seq"] != count or event["previous"] != previous or fingerprint(event) != digest:
                    raise ValueError("Journal integrity check failed")
                state, previous = event["state"], digest
    return manifest, state, previous, count


def _commit(directory, state, kind, previous, count):
    event = {"seq": count + 1, "previous": previous, "kind": kind, "state": state}
    digest = fingerprint(event)
    event["hash"] = digest
    with (Path(directory) / "events.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _atomic_json(Path(directory) / "state.json", state)
    return digest, count + 1


def _position(grid, plan):
    verdict = execute(grid, plan)
    if not verdict.feasible:
        raise ValueError("Corrupt committed path")
    return tuple(verdict.trace[-1]["to"]) if verdict.trace else grid.start


def _tool(grid, state, call, budget, oracle):
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        return "invalid_tool_schema"
    name, args = call["name"], call["arguments"]
    if name not in TOOLS or not isinstance(args, dict):
        return "unknown_tool_or_arguments"
    expected = {"move": {"action"}, "verify_plan": {"plan"}, "submit_plan": {"plan"}, "finish": set()}
    if set(args) != expected[name]:
        return "invalid_tool_arguments"
    if name == "finish":
        state["status"] = "stopped"
        return "agent_finished_without_verified_goal"
    if name == "move":
        action = args["action"]
        if not isinstance(action, str) or action not in ACTIONS:
            return "invalid_action_token"
        plan = " ".join(filter(None, [state["plan"], action]))
        attempts = 1
    else:
        plan = args["plan"]
        if not isinstance(plan, str) or len(plan) > budget.plan_tokens * 16:
            return "invalid_plan_type_or_size"
        plan = " ".join(plan.split())
        attempts = len(plan.split())
        if attempts > budget.plan_tokens:
            return "plan_token_limit"
    if state["action_attempts"] + attempts > budget.action_attempts:
        state["status"] = "budget_exhausted"
        return "action_attempt_budget"
    state["action_attempts"] += attempts
    verdict = oracle.check(grid, plan) if oracle else execute(grid, plan)
    state["candidate"] = plan
    if name == "move" and verdict.feasible or name == "submit_plan" and verdict.valid:
        state["plan"] = plan
        if verdict.valid:
            state["status"] = "success"
    return verdict.label()


def run_session(directory, backend, backend_identity, oracle=None, check=lambda: None):
    """backend.next_call(context) -> {name, arguments}; backend errors are resumable.

    Timeout is cooperative between model calls, not an interruptible hard deadline.
    Every request is durably charged before calling the backend, even on crashes.
    """
    directory = Path(directory)
    with _lock(directory):
        manifest, state, previous, count = read_session(directory)
        if manifest["backend"] != backend_identity:
            raise ValueError("Backend/config/weights changed; create a new trial")
        actual_oracle = {"kind": "official", "sha256": oracle.sha256} if oracle else {"kind": "local"}
        if actual_oracle != manifest["oracle"]:
            raise ValueError("Oracle changed; create a new trial")
        grid, budget = Grid.from_dict(manifest["grid"]), Budgets(**manifest["budgets"])
        if state["status"] in ("success", "stopped", "budget_exhausted"):
            return state
        while True:
            if (state["model_calls"] >= budget.model_calls or state["tool_calls"] >= budget.tool_calls
                    or state["active_seconds"] >= budget.active_seconds):
                state["status"] = "budget_exhausted"
                _commit(directory, state, "stop", previous, count)
                return state
            check()
            state["status"] = "running"
            state["model_calls"] += 1
            previous, count = _commit(directory, state, "model_request", previous, count)
            started = time.monotonic()
            context = {"grid": manifest["grid"], "position": list(_position(grid, state["plan"])),
                       "state": deepcopy(state), "tools": TOOLS}
            try:
                call = backend.next_call(context)
            except Exception as error:
                state["active_seconds"] += time.monotonic() - started
                state["status"] = "backend_error"
                state["feedback"] = type(error).__name__ + ": " + str(error)[:1000]
                _commit(directory, state, "backend_error", previous, count)
                raise
            state["active_seconds"] += time.monotonic() - started
            if state["active_seconds"] >= budget.active_seconds:
                state["status"] = "budget_exhausted"
                _commit(directory, state, "stop", previous, count)
                return state
            check()
            state["tool_calls"] += 1
            before = state["plan"]
            started = time.monotonic()
            feedback = _tool(grid, state, call, budget, oracle)
            state["active_seconds"] += time.monotonic() - started
            state["feedback"] = feedback
            if feedback != "valid" and state["plan"] == before:
                state["invalid_calls"] += 1
            # Bound untrusted outputs before persisting or returning them to a model.
            state["history"].append({"call": repr(call)[:2048], "feedback": feedback})
            state["history"] = state["history"][-budget.context_events:]
            previous, count = _commit(directory, state, "tool_result", previous, count)
            if state["status"] in ("success", "stopped", "budget_exhausted"):
                return state
