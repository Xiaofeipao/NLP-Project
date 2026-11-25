"""Native seq2seq scoring for the frozen coordinate G/C/V tasks."""

from functools import lru_cache
from pathlib import Path
import shutil

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

from .protocol import LABELS, MOVES, put, read


def padded(rows, fill, device):
    width = max(map(len, rows))
    return torch.tensor([list(r) + [fill] * (width - len(r)) for r in rows], device=device)


class Network:
    def __init__(self, folder, device="cuda"):
        folder = Path(folder)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(folder), local_files_only=True, dtype=torch.float32
        ).to(device)
        self.device = self.model.device
        self.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tok.no_padding()
        self.tok.no_truncation()
        config = self.model.config
        self.pad, self.eos = config.pad_token_id, config.eos_token_id
        if config.model_type not in ("bart", "t5"):
            raise ValueError(config.model_type)
        self.bos = config.bos_token_id if config.model_type == "bart" else None
        first = self.model.generation_config.decoder_start_token_id
        self.prefix = (first,) + ((self.bos,) if self.bos is not None else ())
        if config.model_type == "t5":
            assert not getattr(config, "scale_decoder_outputs", False)
            assert self.model.lm_head.weight is not self.model.shared.weight
        self.label_ids = [self.target(x)[0] for x in LABELS]
        if len(set(self.label_ids)) != len(LABELS):
            raise ValueError("Verifier labels share a first content token")
        self.moves = [self.target(a) for a in MOVES]

    @lru_cache(maxsize=512)
    def target(self, text):
        ids = self.tok.encode(text).ids
        assert ids[-1] == self.eos and self.tok.decode(ids) == text
        if self.bos is not None:
            assert ids[0] == self.bos
            ids = ids[1:]
        return tuple(ids)

    def encode(self, texts):
        rows = [r.ids for r in self.tok.encode_batch(texts)]
        if any(len(r) > 512 or r[-1] != self.eos for r in rows):
            raise ValueError("Malformed or oversized prompt")
        return rows

    def logits(self, inputs, choices):
        source = padded(inputs, self.pad, self.device)
        mask = source.ne(self.pad).long()
        counts = [len(opts) for opts in choices]
        targets = padded([a for opts in choices for a in opts], -100, self.device)
        prefix = source.new_tensor(self.prefix).expand(len(targets), -1)
        decoder = torch.cat((prefix, targets[:, :-1].masked_fill(targets[:, :-1] == -100, self.pad)), dim=1)
        if len(targets) == len(source):
            out = self.model(
                input_ids=source, attention_mask=mask, decoder_input_ids=decoder, use_cache=False
            )
        else:
            enc = self.model.get_encoder()(input_ids=source, attention_mask=mask, return_dict=True)
            owners = source.new_tensor([i for i, n in enumerate(counts) for _ in range(n)])
            shared = BaseModelOutput(last_hidden_state=enc.last_hidden_state.index_select(0, owners))
            out = self.model(
                encoder_outputs=shared,
                attention_mask=mask.index_select(0, owners),
                decoder_input_ids=decoder,
                use_cache=False,
            )
        return out.logits[:, len(self.prefix) - 1 :], targets

    def loss(self, inputs, choices, classes):
        logits, targets = self.logits(inputs, choices)
        counts = [len(c) for c in choices]
        seq, nseq, ver, gold = [], [], [], []
        offset = 0
        for n, cls in zip(counts, classes, strict=True):
            if cls is None:
                seq.extend(range(offset, offset + n))
                nseq.append(n)
            else:
                assert n == 1
                ver.append(offset)
                gold.append(cls)
            offset += n
        sequence_loss = verdict_loss = iter(())
        if seq:
            lp = -F.cross_entropy(logits[seq].float().transpose(1, 2), targets[seq], reduction="none").sum(
                dim=1
            )
            sequence_loss = iter([-part.logsumexp(0) for part in lp.split(nseq)])
        if ver:
            scores = logits[ver, 0][:, self.label_ids].float()
            verdict_loss = iter(F.cross_entropy(scores, targets.new_tensor(gold), reduction="none").unbind())
        return torch.stack([next(sequence_loss) if cls is None else next(verdict_loss) for cls in classes])

    @torch.inference_mode()
    def predict(self, task, texts):
        result = []
        for offset in range(0, len(texts), 8):
            group = self.encode(texts[offset : offset + 8])
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                if task == "C":
                    ids = padded(group, self.pad, self.device)
                    # Give BART its fixed BOS up front. Both families have 32 content-token slots.
                    prefix = ids.new_tensor(self.prefix).expand(len(ids), -1).contiguous()
                    raw = self.model.generate(
                        input_ids=ids,
                        attention_mask=ids.ne(self.pad).long(),
                        decoder_input_ids=prefix,
                        max_new_tokens=32,
                        num_beams=1,
                        do_sample=False,
                        no_repeat_ngram_size=0,
                        early_stopping=False,
                        use_cache=True,
                    )
                    for row in raw.tolist():
                        assert tuple(row[: len(self.prefix)]) == self.prefix
                        seq = row[len(self.prefix) :]
                        result.append(self.tok.decode(seq[: seq.index(self.eos)] if self.eos in seq else seq))
                elif task == "V":
                    logits, _ = self.logits(group, [[(self.label_ids[0],)]] * len(group))
                    scores = logits[:, 0, self.label_ids].float()
                    if not torch.isfinite(scores).all():
                        raise ValueError("Nonfinite verifier scores")
                    result.extend(LABELS[i] for i in scores.argmax(1).tolist())
                elif task == "G":
                    logits, tgt = self.logits(group, [self.moves] * len(group))
                    scores = -F.cross_entropy(logits.float().transpose(1, 2), tgt, reduction="none").sum(1)
                    if not torch.isfinite(scores).all():
                        raise ValueError("Nonfinite generator scores")
                    result.extend(
                        [MOVES[i] for i in ranks]
                        for ranks in scores.reshape(-1, 4)
                        .argsort(dim=1, descending=True, stable=True)
                        .tolist()
                    )
                else:
                    raise ValueError(task)
        return result

    def save(self, base, folder):
        self.model.save_pretrained(folder)
        if self.model.config.model_type == "t5":
            config = read(folder / "config.json")
            config.update(tie_word_embeddings=False, scale_decoder_outputs=False)
            put(folder / "config.json", config)
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "spiece.model",
            "merges.txt",
            "vocab.json",
            "added_tokens.json",
        ):
            if (base / name).exists():
                shutil.copyfile(base / name, folder / name)
