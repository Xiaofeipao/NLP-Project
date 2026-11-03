"""Shared train/evaluation templates; no hidden shortest-path hint at inference."""


def checked_prompt_format(value):
    if value not in ("vanilla", "structured"):
        raise ValueError(f"Unsupported prompt format: {value}")
    return value


def protocol_model_config(config, protocol):
    style = checked_prompt_format(protocol.get("prompt_format", "vanilla"))
    if "prompt_format" in config and config["prompt_format"] != style:
        raise ValueError("Config and protocol prompt formats disagree")
    return {**config, "prompt_format": style}


def prompt(task, grid, candidate="", feedback="", *, input_format="vanilla"):
    checked_prompt_format(input_format)
    context = grid.description()
    if input_format == "structured":
        # Observable problem state only: no reference actions or executor labels.
        obstacles = " ".join(f"({r},{c})" for r, c in sorted(set(grid.obstacles))) or "none"
        context = (f"Grid: {grid.rows}x{grid.cols}\nCoordinates: zero-based (row,column). "
                   "Up decreases row; right increases column.\n"
                   f"Start: ({grid.start[0]},{grid.start[1]})\n"
                   f"Goal: ({grid.goal[0]},{grid.goal[1]})\nObstacles: {obstacles}")
    if task == "generation":
        return f"Plan a path. {context}\nReturn only space-separated up down left right actions."
    if task == "verification":
        return (f"Verify a path. {context}\nCandidate: {candidate}\n"
                "Return valid or invalid | step=<1-based index> | reason=<error_type>. "
                "Reasons: obstacle_collision, out_of_bounds, invalid_action_token, goal_not_reached. "
                "An unfinished plan fails at the action count plus one.")
    if task == "repair":
        return (f"Repair a path. {context}\nCandidate: {candidate}\nFeedback: {feedback}\n"
                "Return only the corrected space-separated up down left right actions.")
    raise ValueError(task)


def format_training_rows(rows, input_format):
    """Reformat inputs only; exact IDs, quotas and supervision stay unchanged."""
    from .core import Grid
    checked_prompt_format(input_format)
    return [{**row, "input": prompt(row["task"], Grid.from_dict(row["grid"]),
                                    row.get("candidate", ""), row.get("label", ""),
                                    input_format=input_format)} for row in rows]


def check_checkpoint_prompt_format(directory, expected):
    import json
    from pathlib import Path
    checked_prompt_format(expected)
    path = Path(directory) / "prompt_format.json"
    actual = json.loads(path.read_text(encoding="utf-8"))["prompt_format"] if path.exists() else "vanilla"
    if checked_prompt_format(actual) != expected:
        raise ValueError("Checkpoint prompt format differs; train a matching base first")
