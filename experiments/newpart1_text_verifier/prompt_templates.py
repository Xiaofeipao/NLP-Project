"""Versioned GitHub and compact local verifier prompts."""

import json

def step_prompt(grid, position, action):
    return f'Verify one step in a {grid.rows} by {grid.cols} grid. Coordinates are zero-based (row, column). Up decreases row; right increases column.\nCurrent position: {tuple(position)}. Goal: {grid.goal}. Obstacles: {list(grid.obstacles)}.\nCandidate action: {json.dumps(action)}. A null action means stop without moving.\nReturn valid if the action is legal and ends at the goal (or stop at the goal). For an illegal action, return invalid | step=1 | reason=<error_type>, where the type is invalid_action_token, out_of_bounds, or obstacle_collision. If the action is legal but the goal has not been reached, return invalid | step=2 | reason=goal_not_reached. Use step=1 if stopping without moving.'

def verification_prompt(grid, position, action, style='compact'):
    if style == 'github':
        return step_prompt(grid, position, action)
    fields = f"Grid: {grid.rows}x{grid.cols}\nPosition: {json.dumps(list(position), separators=(',', ':'))}\nGoal: {json.dumps(list(grid.goal), separators=(',', ':'))}\nObstacles: {json.dumps([list(p) for p in sorted(grid.obstacles)], separators=(',', ':'))}\nAction: {json.dumps(action)}\n"
    if style == 'compact':
        return 'Verify one action. Zero-based [row,column]; up decreases row, right increases column.\n' + fields + 'null means stop. Reply valid only for legal arrival or stopping at goal.\nOtherwise reply invalid | step=N | reason=R.\nR: invalid_action_token (unknown), out_of_bounds (outside), obstacle_collision (blocked), goal_not_reached (legal but unfinished).\nN=1 for illegal action or null; N=2 for legal unfinished move.'
    return 'Verify exactly one action using zero-based [row,column].\nup=[-1,0]; down=[1,0]; left=[0,-1]; right=[0,1]. null means stop.\n' + fields + 'Check in order: unknown action; next row/column outside grid; obstacle; arrival.\nRows range from 0 to rows-1. Columns range from 0 to columns-1.\nReturn exactly one line:\nvalid\ninvalid | step=1 | reason=invalid_action_token\ninvalid | step=1 | reason=out_of_bounds\ninvalid | step=1 | reason=obstacle_collision\nIf legal but not at goal: invalid | step=2 | reason=goal_not_reached\nIf null and not at goal: invalid | step=1 | reason=goal_not_reached\nvalid applies only to legal arrival or null at goal.'
