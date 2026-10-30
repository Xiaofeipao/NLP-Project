"""Normalize accumulated gradients by the units used by the loss.

Pattern reference: https://huggingface.co/blog/gradient_accumulation
Token-mean losses need the entire window's non-padding token count, rather
than an unweighted mean of microbatch means. Example-mean objectives instead
use the number of examples. This implementation is original project code.
"""
from itertools import islice

ACCUMULATION_NORMALIZATION = "loss_units_v1"


def accumulated_batches(loader, steps, reduction="tokens"):
    """Yield (index, CPU batch, loss scale, update) for each microbatch.

    Buffer only one accumulation window on CPU; callers move one batch at a
    time to the GPU. Loss must be a mean over the specified reduction units.
    The last incomplete window uses its actual token/example count.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("Accumulation steps must be a positive integer")
    if reduction not in ("tokens", "examples"):
        raise ValueError("Unknown loss reduction")
    iterator = iter(loader)
    index = 0
    while window := list(islice(iterator, steps)):
        counts = [(batch["labels"] != -100).sum().item() if reduction == "tokens"
                  else len(batch["labels"]) for batch in window]
        if any(count <= 0 for count in counts):
            raise ValueError("Every microbatch needs supervised loss units")
        total = sum(counts)
        for offset, (batch, count) in enumerate(zip(window, counts)):
            yield index, batch, count / total, offset == len(window) - 1
            index += 1
