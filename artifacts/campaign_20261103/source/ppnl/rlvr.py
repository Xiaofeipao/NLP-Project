"""Executor rewards and on-policy REINFORCE helpers (not human feedback)."""
from .core import ACTIONS, execute


def plan_reward(grid, plan):
    verdict = execute(grid, plan)
    if verdict.valid:
        return 2.0 + 1.0 / (1 + len(plan.split()))
    # Terminal progress is shaping, not a success metric. Illegal moves always
    # receive a penalty, and padding/repeated prose cannot earn prefix credit.
    pos = grid.start
    for token in plan.split():
        if token not in ACTIONS:
            break
        dr, dc = ACTIONS[token]
        nxt = (pos[0] + dr, pos[1] + dc)
        if not grid.inside(nxt) or nxt in grid.obstacles:
            break
        pos = nxt
    distance = lambda p: abs(p[0] - grid.goal[0]) + abs(p[1] - grid.goal[1])
    progress = (distance(grid.start) - distance(pos)) / max(1, distance(grid.start))
    return .5 * progress - (0 if verdict.feasible else 1) - .01 * len(plan.split())


def leave_one_out(rewards):
    if len(rewards) < 2:
        raise ValueError("At least two samples required")
    return [(r - (sum(rewards) - r) / (len(rewards) - 1)) for r in rewards]


def sequence_objective(logp, reference_logp, mask, advantages, beta):
    """One on-policy update; sampled KL estimator against frozen base, no PPO."""
    import torch
    delta = reference_logp - logp
    kl = torch.exp(delta.clamp(-20, 20)) - 1 - delta
    per_token = -logp * advantages[:, None] + beta * kl
    return ((per_token * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()


def group_advantages(rewards):
    """Outcome GRPO: within-prompt population standard deviation normalization."""
    import torch
    values = torch.as_tensor(rewards, dtype=torch.float32)
    if values.numel() < 2:
        raise ValueError("GRPO needs at least two completions")
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-4)


def completion_mask(labels, eos_id, pad_id):
    """Include the first EOS, exclude post-EOS padding (including BART EOS start)."""
    eos = labels.eq(eos_id)
    before_or_first = eos.cumsum(1) - eos.to(labels.dtype) == 0
    return before_or_first & labels.ne(pad_id)


def grpo_objective(logp, old_logp, reference_logp, mask, advantages, beta, epsilon=.2):
    """DeepSeekMath outcome-GRPO token ratios, clipping and reference KL.

    old_logp must be captured before ALL updates on the sampled rollout.
    Sequence means reproduce the original length normalization, not later DAPO.
    """
    import torch
    ratio = torch.exp(logp - old_logp.detach())
    advantage = advantages.detach()[:, None]
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1-epsilon, 1+epsilon) * advantage)
    delta = reference_logp.detach() - logp
    kl = torch.exp(delta.clamp(-20, 20)) - 1 - delta
    return ((-surrogate + beta * kl) * mask).sum(1).div(mask.sum(1).clamp_min(1)).mean()
