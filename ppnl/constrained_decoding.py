"""Bounded whole-plan action decoding; the scorer never sees reference plans."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .core import ACTIONS, Grid

TOKENS = (*ACTIONS, "<eos>")
ACTION_ORDER = {name: index for index, name in enumerate(ACTIONS)}


@dataclass(frozen=True)
class DecodeSpec:
    beam_width: int = 4
    max_actions: int = 36
    state_constraints: bool = True

    def __post_init__(self):
        if not 1 <= self.beam_width <= 16 or not 1 <= self.max_actions <= 128:
            raise ValueError("Invalid action beam budget")


@dataclass(frozen=True)
class BeamState:
    actions: tuple[str, ...]
    position: tuple[int, int]
    visited: frozenset[tuple[int, int]]
    logprob: float = 0.0


@dataclass(frozen=True)
class DecodeResult:
    plan: str
    stop_reason: str
    completed: bool
    model_logprob: float
    action_count: int
    forward_batches: int
    decoder_row_steps: int
    pruned_illegal: int
    pruned_revisit: int
    dead_beams: int


def decode_actions(grid: Grid, score, spec: DecodeSpec = DecodeSpec(), check=lambda: None):
    """Score all five next tokens before masking; never renormalize legal actions.

    ``score(prefixes)`` returns one {action/eos: full-vocabulary log-prob}
    mapping per prefix. It may batch the prefixes in one model forward.
    """
    active = [BeamState((), grid.start, frozenset((grid.start,)))]
    complete = []
    last_partial = active[0]
    batches = rows = illegal = revisits = dead = 0

    def rank(beam):
        return (-beam.logprob, tuple(ACTION_ORDER[action] for action in beam.actions))

    for _ in range(spec.max_actions + 1):
        if not active:
            break
        check()
        values = score([b.actions for b in active])
        if len(values) != len(active):
            raise ValueError("Scorer returned the wrong number of rows")
        batches += 1
        rows += len(active)
        next_active = []
        for beam, row in zip(active, values):
            if not isinstance(row, dict):
                raise TypeError("Action scores must be mappings")
            expanded = False
            for token in TOKENS:
                value = row.get(token, -math.inf)
                if not isinstance(value, (int, float)) or math.isnan(value):
                    raise ValueError("Action scores must be numeric")
                if not math.isfinite(value):
                    continue
                if token == "<eos>":
                    if not spec.state_constraints or beam.position == grid.goal:
                        complete.append(BeamState(beam.actions, beam.position, beam.visited,
                                                  beam.logprob + value))
                        expanded = True
                    continue
                if len(beam.actions) >= spec.max_actions:
                    continue
                dr, dc = ACTIONS[token]
                position = (beam.position[0] + dr, beam.position[1] + dc)
                if spec.state_constraints:
                    if (beam.position == grid.goal or not grid.inside(position)
                            or position in grid.obstacles):
                        illegal += 1
                        continue
                    if position in beam.visited:
                        revisits += 1
                        continue
                next_active.append(BeamState(beam.actions + (token,), position,
                                             beam.visited | {position}, beam.logprob + value))
                expanded = True
            if not expanded:
                dead += 1
        complete = sorted(complete, key=rank)[:spec.beam_width]
        active = sorted(next_active, key=rank)[:spec.beam_width]
        if active:
            last_partial = active[0]
    winner = complete[0] if complete else last_partial
    reason = ("complete" if complete else "action_budget" if active or
              len(last_partial.actions) >= spec.max_actions else "dead_end")
    return DecodeResult(" ".join(winner.actions), reason, bool(complete), winner.logprob,
                        len(winner.actions), batches, rows, illegal, revisits, dead)
