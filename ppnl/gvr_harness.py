"""Journaled whole-plan Generate/Verify/Repair workflow."""
from __future__ import annotations

from pathlib import Path
import time

from .core import Grid, execute, parse_verdict
from .harness import _commit, _lock, create_session, read_session

MODES = {"direct": 0, "self": 1, "executor_one": 1, "executor_three": 3}


def create_gvr_session(directory, grid, mode, identity, model_calls=8, active_seconds=300.0):
    if mode not in MODES or model_calls < 1 or active_seconds <= 0:
        raise ValueError("Invalid GVR protocol")
    # Reuse the existing integrity checked manifest and journal without the
    # move/submit tool, whose truth-dependent stop is unsuitable here.
    from .harness import Budgets
    backend = {"kind": "gvr", "mode": mode, "model": identity,
               "max_repairs": MODES[mode], "model_calls": model_calls,
               "active_seconds": active_seconds}
    return create_session(directory, grid, Budgets(model_calls=model_calls, active_seconds=active_seconds),
                          backend, {"kind": "local-scoring-only"})


def run_gvr_session(directory, model, identity, check=lambda: None):
    directory = Path(directory)
    with _lock(directory):
        manifest, state, previous, count = read_session(directory)
        spec = manifest["backend"]
        if spec.get("kind") != "gvr" or spec["model"] != identity:
            raise ValueError("GVR model/base/adapter identity changed")
        mode = spec["mode"]
        grid = Grid.from_dict(manifest["grid"])
        state.setdefault("phase", "generate")
        state.setdefault("repairs", 0)
        state.setdefault("candidate_history", [])
        state.setdefault("feedback_source", None)
        state.setdefault("stop_reason", None)
        state.setdefault("executor_calls", 0)
        state.setdefault("raw_feedback", "")
        state.setdefault("input_tokens", 0)
        state.setdefault("output_tokens", 0)
        if state["status"] in ("stopped", "budget_exhausted"):
            return state

        def commit(kind):
            nonlocal previous, count
            previous, count = _commit(directory, state, kind, previous, count)

        while True:
            if state["model_calls"] >= spec["model_calls"] or state["active_seconds"] >= spec["active_seconds"]:
                state.update(status="budget_exhausted", stop_reason="budget_exhausted")
                commit("stop")
                return state
            phase = state["phase"]
            if phase == "feedback" and mode.startswith("executor"):
                # True diagnostics are available only in this explicit branch.
                feedback = execute(grid, state["candidate"]).label()
                state["executor_calls"] += 1
                state["raw_feedback"] = feedback
                state["feedback_source"] = "executor"
                state["phase"] = "decide"
                commit("executor_feedback")
                continue
            if phase == "decide":
                parsed = parse_verdict(state["raw_feedback"])
                if parsed == (True, None, None):
                    state.update(status="stopped", stop_reason="feedback_valid")
                elif state["candidate"] in state["candidate_history"][:-1]:
                    state.update(status="stopped", stop_reason="repeated_candidate")
                elif state["repairs"] >= spec["max_repairs"]:
                    state.update(status="stopped", stop_reason="repair_limit")
                else:
                    state["phase"] = "repair"
                commit("decision")
                if state["status"] == "stopped":
                    return state
                continue
            if phase == "done":
                state.update(status="stopped", stop_reason="direct_complete")
                commit("stop")
                return state
            task = {"generate": "generation", "feedback": "verification", "repair": "repair"}[phase]
            check()
            state["model_calls"] += 1  # Durable charge before any model request.
            state["status"] = "running"
            commit("model_request")
            started = time.monotonic()
            try:
                output = model.predict(task, grid, state["candidate"], state["raw_feedback"], check)
                if not isinstance(output, str):
                    raise TypeError("Model response must be text")
            except Exception:
                state["active_seconds"] += time.monotonic() - started
                state["status"] = "backend_error"
                commit("backend_error")
                raise
            state["active_seconds"] += time.monotonic() - started
            usage = getattr(model, "last_usage", {})
            state["input_tokens"] += usage.get("input_tokens", 0)
            state["output_tokens"] += usage.get("output_tokens", 0)
            if task == "verification":
                state["raw_feedback"] = output[:2048]
                state["feedback_source"] = "model"
                state["phase"] = "decide"
                state["history"].append({"task": task, "raw_feedback": state["raw_feedback"]})
            else:
                state["candidate"] = output[:2048]
                state["candidate_history"].append(state["candidate"])
                if task == "repair":
                    state["repairs"] += 1
                state["phase"] = "done" if mode == "direct" else "feedback"
                state["history"].append({"task": task, "candidate": state["candidate"]})
            commit("model_response")


def score_gvr_session(directory, oracle=None):
    """Score only the final candidate; never feed this result into the workflow."""
    manifest, state, _, _ = read_session(directory)
    grid = Grid.from_dict(manifest["grid"])
    verdict = oracle.check(grid, state["candidate"]) if oracle else execute(grid, state["candidate"])
    from .core import shortest_path
    best = shortest_path(grid)
    return {"candidate": state["candidate"], "success": verdict.valid, "feasible": verdict.feasible,
            "optimal": verdict.valid and len(state["candidate"].split()) == len(best),
            "truth": verdict.label(), "model_calls": state["model_calls"],
            "executor_calls": state["executor_calls"], "repairs": state["repairs"],
            "input_tokens": state["input_tokens"], "output_tokens": state["output_tokens"],
            "active_seconds": state["active_seconds"],
            "stop_reason": state["stop_reason"], "feedback_source": state["feedback_source"]}
