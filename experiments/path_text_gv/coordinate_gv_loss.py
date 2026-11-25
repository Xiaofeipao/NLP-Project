"""Full answer-set likelihood for G and one restricted first-position decision for V."""
import torch
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput

from objectives import pad_sequences


def loss_terms(logits, targets, choice_counts, v_classes, verdict_ids):
    g_options, g_counts, v_options, v_gold = [], [], [], []
    begin = 0
    for n, cls in zip(choice_counts, v_classes, strict=True):
        if cls is None:
            g_options.extend(range(begin, begin + n))
            g_counts.append(n)
        else:
            if n != 1:
                raise ValueError("Expected one verifier target")
            v_options.append(begin)
            v_gold.append(cls)
        begin += n
    if begin != len(logits):
        raise ValueError("Decoder options do not align with samples")
    g_losses, v_losses = iter(()), iter(())
    if g_options:
        logp = -F.cross_entropy(logits[g_options].float().transpose(1, 2), targets[g_options],
                                ignore_index=-100, reduction="none").sum(dim=1)
        g_losses = iter(-torch.logsumexp(s, dim=0) for s in logp.split(g_counts))
    if v_options:
        scores = logits[v_options, 0][:, verdict_ids].float()
        v_losses = iter(F.cross_entropy(scores, targets.new_tensor(v_gold), reduction="none").unbind())
    return torch.stack([next(g_losses) if cls is None else next(v_losses) for cls in v_classes])


def view_losses(model, encoded, choices, v_classes, verdict_ids):
    source = pad_sequences(encoded, 0, model.device)
    attention = source.ne(0).long()
    counts = [len(answers) for answers in choices]
    targets = pad_sequences([ids for answers in choices for ids in answers], -100, model.device)
    decoder = model._shift_right(targets)
    if len(targets) == len(source):
        output = model(input_ids=source, attention_mask=attention, decoder_input_ids=decoder, use_cache=False)
    else:
        encoder = model.get_encoder()(input_ids=source, attention_mask=attention, return_dict=True)
        owners = source.new_tensor([i for i, count in enumerate(counts) for _ in range(count)])
        shared = BaseModelOutput(last_hidden_state=encoder.last_hidden_state.index_select(0, owners))
        output = model(encoder_outputs=shared, attention_mask=attention.index_select(0, owners),
                       decoder_input_ids=decoder, use_cache=False)
    return loss_terms(output.logits, targets, counts, v_classes, verdict_ids)


def verdict_scores(model, encoded, verdict_ids):
    source = pad_sequences(encoded, 0, model.device)
    start = source.new_full((len(source), 1), model.config.decoder_start_token_id)
    output = model(input_ids=source, attention_mask=source.ne(0).long(), decoder_input_ids=start, use_cache=False)
    return output.logits[:, 0, verdict_ids].float()
