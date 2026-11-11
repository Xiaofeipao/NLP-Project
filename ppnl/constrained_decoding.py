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
    backtrack: bool = False
    max_score_rows: int | None = None

    def __post_init__(self):
        if (type(self.beam_width) is not int or type(self.max_actions) is not int
                or not 1 <= self.beam_width <= 16 or not 1 <= self.max_actions <= 128):
            raise ValueError("Invalid action beam budget")
        if type(self.backtrack) is not bool or type(self.state_constraints) is not bool:
            raise ValueError("Decoder switches must be booleans")
        if self.max_score_rows is not None and (type(self.max_score_rows) is not int or not 1 <= self.max_score_rows <= 100000):
            raise ValueError("Invalid scorer row budget")
        if self.backtrack and (not self.state_constraints or self.max_score_rows is None):
            raise ValueError("Backtracking requires state constraints and an explicit scorer row budget")


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
    deferred = []
    row_budget_hit = False

    def rank(beam):
        return (-beam.logprob, tuple(ACTION_ORDER[action] for action in beam.actions))

    for _ in range(spec.max_score_rows if spec.backtrack else spec.max_actions + 1):
        if not active and deferred:
            # Recover siblings from the most recent abandoned depth. Each batch
            # has equal-length prefixes, as required by the T5 scorer.
            siblings = deferred.pop()
            active = siblings[:spec.beam_width]
            if len(siblings) > spec.beam_width:
                deferred.append(siblings[spec.beam_width:])
        if not active:
            break
        if spec.max_score_rows is not None:
            remaining = spec.max_score_rows - rows
            if remaining <= 0:
                row_budget_hit = True
                break
            active = active[:remaining]
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
        ordered = sorted(next_active, key=rank)
        if spec.backtrack and len(ordered) > spec.beam_width:
            deferred.append(ordered[spec.beam_width:])
        active = ordered[:spec.beam_width]
        if active:
            last_partial = active[0]
    winner = complete[0] if complete else last_partial
    row_budget_hit = row_budget_hit or (spec.max_score_rows is not None and rows >= spec.max_score_rows)
    reason = ("complete" if complete else "scorer_budget" if row_budget_hit else "action_budget" if active or
              len(last_partial.actions) >= spec.max_actions else "dead_end")
    return DecodeResult(" ".join(winner.actions), reason, bool(complete), winner.logprob,
                        len(winner.actions), batches, rows, illegal, revisits, dead)
