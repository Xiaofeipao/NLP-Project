"""Shared train/evaluation templates; no hidden shortest-path hint at inference."""


def prompt(task, grid, candidate="", feedback=""):
    context = grid.description()
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
