"""Train and decode V with cross-entropy over the five first-token logits."""
import torch
from torch.nn import functional as F

from ppnl.modeling import Seq2Seq, optimal_action_loss
from . import g_probes, model_io
from .single import LABELS


def example_losses(logits, first_targets, allowed, action_ids, verdict_ids):
    is_g = allowed.any(dim=1)
    g_loss = optimal_action_loss(logits[:, action_ids], allowed | ~is_g[:, None])
    targets = (first_targets[:, None] == first_targets.new_tensor(verdict_ids)).long().argmax(dim=1)
    v_loss = F.cross_entropy(logits[:, verdict_ids].float(), targets, reduction="none")
    return torch.where(is_g, g_loss, v_loss)


def mixed_loss(logits, first_targets, allowed, action_ids, verdict_ids):
    return example_losses(logits, first_targets, allowed, action_ids, verdict_ids).mean()


class FirstTokenRunner(Seq2Seq):
    verifier_loss = g_probes.FIRST_TOKEN_LOSS

    def load_model(self, checkpoint, **kwargs):
        model = super().load_model(checkpoint, **kwargs)
        targets = self.tokenizer(text_target=list(LABELS), add_special_tokens=False)["input_ids"]
        if not all(targets):
            raise ValueError("Each V label must contain a token")
        self.verdict_ids = [t[0] for t in targets]
        if len(set(self.verdict_ids)) != len(LABELS) or set(self.verdict_ids) & set(self.tokenizer.all_special_ids):
            raise ValueError("Each V label must start with a unique non-special token")
        mode = getattr(model.config, "ppnl_verifier_loss", None)
        if model.config.model_type not in ("t5", "bart") or mode not in (None, self.verifier_loss):
            raise ValueError("First-token V requires T5, BART, or a matching saved checkpoint")
        self.verdict_bos = self.tokenizer.bos_token_id if model.config.model_type == "bart" else None
        if model.config.model_type == "bart" and self.verdict_bos != model.config.bos_token_id:
            raise ValueError("BART model and tokenizer disagree on BOS")
        if mode is not None and (model.config.ppnl_verifier_labels != list(LABELS)
                                 or model.config.ppnl_verifier_token_ids != self.verdict_ids):
            raise ValueError("Saved V labels or token IDs differ from the tokenizer")
        return model

    def prepare_training(self, dataset):
        source = model_io.read_split(dataset, "train", version=g_probes.VERSION,
                                     fingerprint=g_probes.template_hash())
        return self.prepare_rows(source)

    def prepare_rows(self, source):
        rows = model_io.encode_training_rows(self, source, fingerprint=g_probes.template_hash())
        label_ids = dict(zip(LABELS, self.verdict_ids, strict=True))
        for raw, feature in zip(source, rows, strict=True):
            if raw["mode"] != "v_step":
                continue
            if self.verdict_bos is not None:
                if feature["labels"][0] != self.verdict_bos:
                    raise ValueError("BART V target must start with BOS")
                feature["labels"] = feature["labels"][1:]
            if feature["labels"][0] != label_ids[raw["target"]]:
                raise ValueError(f"First token differs from the V target for {raw['id']}")
        self.model.config.ppnl_verifier_loss = self.verifier_loss
        self.model.config.ppnl_verifier_labels = list(LABELS)
        self.model.config.ppnl_verifier_token_ids = self.verdict_ids
        self.model.config.ppnl_verifier_bos = self.verdict_bos
        return rows

    def first_logits(self, inputs, is_g=None):
        ids = inputs["input_ids"]
        start = ids.new_full((len(ids), 1), self.model.config.decoder_start_token_id)
        if self.verdict_bos is None:
            return self.model(input_ids=ids, attention_mask=inputs["attention_mask"],
                              decoder_input_ids=start, use_cache=False).logits[:, 0]
        prefix = torch.cat((start, torch.full_like(start, self.verdict_bos)), dim=1)
        logits = self.model(input_ids=ids, attention_mask=inputs["attention_mask"],
                            decoder_input_ids=prefix, use_cache=False).logits
        if is_g is None:
            return logits[:, 1]
        return logits[torch.arange(len(ids), device=ids.device), (~is_g).long()]

    def batch_losses(self, batch):
        batch = batch.to(self.device)
        allowed = batch["policy_actions"]
        return example_losses(self.first_logits(batch, allowed.any(dim=1)), batch["labels"][:, 0], allowed,
                              self.action_ids, self.verdict_ids)

    def batch_loss(self, batch):
        return self.batch_losses(batch).mean()

    def predict(self, inputs, check=lambda: None, *, allow_valid=None):
        """Restrict only the valid logit on rows whose candidate cannot reach the goal."""
        if allow_valid is not None and len(allow_valid) != len(inputs):
            raise ValueError("Expected one goal check per verifier input")
        self.model.eval()
        results = []
        size = self.config["training"]["eval_batch_size"]
        for offset in range(0, len(inputs), size):
            check()
            batch = self.tokenizer(inputs[offset:offset + size], padding=True, truncation=False,
                                   return_tensors="pt").to(self.device)
            if batch.input_ids.shape[1] > self.input_limit:
                raise ValueError("Verifier input exceeds configured limit")
            with torch.inference_mode(), self.autocast():
                scores = self.first_logits(batch)[:, self.verdict_ids]
                if allow_valid is not None:
                    allowed = torch.tensor(allow_valid[offset:offset + size], device=scores.device, dtype=torch.bool)
                    scores[:, LABELS.index("valid")].masked_fill_(~allowed, -torch.inf)
                indices = scores.argmax(dim=-1).tolist()
            results.extend(LABELS[i] for i in indices)
        return results
