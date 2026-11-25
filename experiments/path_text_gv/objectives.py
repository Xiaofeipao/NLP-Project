"""Losses for a text-to-text model. The only model inputs are token IDs for frozen prompts."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import math
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers.modeling_outputs import BaseModelOutput

from texts import CLASSES, INVALID, parse


class Codec:
    def __init__(self, path):
        self.tok = Tokenizer.from_file(str(path))
        self.tok.no_padding()
        self.tok.no_truncation()
        self.pad = self.tok.token_to_id("<pad>")
        self.eos = self.tok.token_to_id("</s>")
        assert (self.pad, self.eos) == (0, 1)

    @lru_cache(maxsize=1024)
    def target(self, text):
        enc = self.tok.encode(text)
        assert enc.ids[-1] == self.eos and self.decode(enc.ids) == text
        return tuple(enc.ids)

    @lru_cache(maxsize=256)
    def fields(self, text):
        """The worst token of each present field supplies its training loss, including EOS as syntax."""
        enc = self.tok.encode(text)
        match = INVALID.fullmatch(text)
        if text != "valid" and not match:
            raise ValueError(text)
        spans = [(0, 5 if text == "valid" else 7, 0)]
        if match:
            spans.extend([(*match.span(1), 1), (*match.span(2), 2)])
        groups = []
        for left, right in enc.offsets:
            groups.append(next((k for lo, hi, k in spans if min(right, hi) > max(left, lo)), 3))
        assert groups[-1] == 3 and set(groups) == ({0, 1, 2, 3} if match else {0, 3})
        return groups

    def decode(self, ids):
        return self.tok.decode(list(ids), skip_special_tokens=True)

    def encode_rows(self, rows, limit=512):
        result = []
        for start in range(0, len(rows), 128):
            enc = self.tok.encode_batch([r["text"] for r in rows[start:start+128]])
            result.extend(x.ids for x in enc)
        if any(len(ids) > limit or ids[-1] != self.eos for ids in result):
            raise ValueError("Input exceeds the context budget or lacks EOS")
        return result


def pad_sequences(items, value, device):
    width = max(map(len, items))
    return torch.tensor([list(x) + [value] * (width-len(x)) for x in items], device=device)


def epoch_weights(rows):
    """Per-view masses summing to N, so a shuffled mean estimates the fixed epoch objective."""
    n = len(rows)
    g = sum(r["task"] == "G" for r in rows)
    cells = Counter((r["label"], r["action"]) for r in rows if r["task"] == "V")
    dirs = Counter(label for label, _ in cells)
    assert set(dirs) == set(CLASSES)
    weights = [0.5*n/g if r["task"]=="G" else
               0.5*n/(len(CLASSES)*dirs[r["label"]]*cells[r["label"],r["action"]]) for r in rows]
    total = Counter()
    for r, w in zip(rows, weights, strict=True):
        total[r["label"]] += w/n
    assert math.isclose(total["G"], 0.5) and all(math.isclose(total[k], 0.1) for k in CLASSES)
    return weights, dict(total)


def preferences(rows, seed):
    """Only other *existing frozen train answer texts* are eligible rejections."""
    bank = sorted({r["answers"][0] for r in rows if r["task"]=="V"})
    by_field = {s:parse(s) for s in bank}
    uses, breakdown, out = Counter(), Counter(), {}
    rng = random.Random(seed)
    for r in rows:
        if r["task"] != "V":
            continue
        gold = r["answers"][0]
        label, step = by_field[gold]
        choices = defaultdict(list)
        for other, (other_label, other_step) in by_field.items():
            if other == gold:
                continue
            if (label=="valid") != (other_label=="valid"):
                kind = "status"
            elif label==other_label:
                kind = "step"
            elif step==other_step:
                kind = "reason"
            else:
                continue
            choices[kind].append(other)
        kinds = sorted(choices)
        kind = kinds[uses[gold] % len(kinds)]
        uses[gold] += 1
        breakdown[kind] += 1
        out[r["id"]] = rng.choice(choices[kind])
    return out, breakdown


@dataclass
class Options:
    ids: list
    fields: list | None = None
    risks: list | None = None
    reference_gap: float = 0.0


def score(model, encoded, choices):
    device = model.device
    source = pad_sequences(encoded, 0, device)
    attention = source.ne(0).long()
    owner = torch.tensor([i for i, opts in enumerate(choices) for _ in opts.ids], device=device)
    targets = pad_sequences([ids for opts in choices for ids in opts.ids], -100, device)
    decoder = model._shift_right(targets)
    encoder = model.get_encoder()(input_ids=source, attention_mask=attention, return_dict=True)
    logits = model(encoder_outputs=BaseModelOutput(last_hidden_state=encoder.last_hidden_state.index_select(0, owner)),
                   attention_mask=attention.index_select(0, owner), decoder_input_ids=decoder,
                   use_cache=False, return_dict=True).logits
    ce = F.cross_entropy(logits.float().transpose(1, 2), targets, ignore_index=-100, reduction="none")
    return -ce.sum(dim=1), ce


def compute(model, encoded, rows, choices, mode, beta, alpha):
    sums, token_nll = score(model, encoded, choices)
    losses = []
    offset = 0
    for r, opts in zip(rows, choices, strict=True):
        end = offset + len(opts.ids)
        logp = sums[offset:end]
        if r["task"] == "G":
            loss = -torch.logsumexp(logp, dim=0)
        elif mode == "field":
            weights = (0.5, 0.2, 0.2, 0.1) if r["label"] != "valid" else (0.5, 0, 0, 0.5)
            parts = [w * token_nll[offset, [j for j, f in enumerate(opts.fields) if f==k]].max()
                     for k, w in enumerate(weights) if w]
            loss = torch.stack(parts).sum()
        elif mode == "dpo":
            assert len(opts.ids) == 2
            loss = -F.logsigmoid(beta * (logp[0]-logp[1]-opts.reference_gap))
        elif mode == "mrt":
            risks = logp.new_tensor(opts.risks)
            loss = (torch.softmax(alpha * logp, dim=0) * risks).sum()
        else:
            raise ValueError(mode)
        losses.append(loss)
        offset = end
    assert offset == sums.numel()
    return torch.stack(losses)
