"""Partial verifier checks; never replace a model diagnosis with an unrequested check."""
from ppnl.core import ACTIONS

RULES = ('none', 'vocabulary', 'terminal', 'bounds', 'collision', 'safety', 'safety_terminal')


def enhance(grid, candidate, prediction, rule):
    if rule not in RULES:
        raise ValueError(rule)
    if rule == 'none':
        return prediction
    pos = grid.start
    actions = candidate.split()
    for step, action in enumerate(actions, 1):
        if action not in ACTIONS:
            reason = 'invalid_action_token'
        else:
            dr, dc = ACTIONS[action]
            nxt = pos[0]+dr, pos[1]+dc
            reason = ('out_of_bounds' if not grid.inside(nxt) else
                      'obstacle_collision' if nxt in grid.obstacles else None)
        if reason:
            # Stop at the first unsafe transition. Never diagnose a later token.
            allowed = {'invalid_action_token'}
            if rule in ('bounds', 'safety', 'safety_terminal'):
                allowed.add('out_of_bounds')
            if rule in ('collision', 'safety', 'safety_terminal'):
                allowed.add('obstacle_collision')
            return f'invalid | step={step} | reason={reason}' if reason in allowed else prediction
        pos = nxt
    if rule in ('terminal', 'safety_terminal') and pos != grid.goal:
        return f'invalid | step={len(actions)+1} | reason=goal_not_reached'
    # Even the strongest hybrid keeps false rejections of genuinely valid plans.
    return prediction
