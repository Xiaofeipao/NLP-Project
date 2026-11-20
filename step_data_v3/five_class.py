"""Independent five-way verifier for the balanced, single-step G/V experiment."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from ppnl.modeling import Seq2Seq, optimal_action_loss
from ppnl.policy import NAMES
from . import g_probes, model_io


# Indices 0 through 4 are A through E, including in saved checkpoints.
CLASS_LABELS = ("valid", "goal_not_reached", "obstacle_collision", "out_of_bounds", "invalid_action_token")
HEAD = "encoder_masked_mean_linear"
VERIFIER_LOSS = "five_class_cross_entropy"


class T5WithStepClassifier(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.verifier_head = nn.Linear(config.d_model, len(CLASS_LABELS))
        self.post_init()

    def classify(self, hidden, attention_mask):
        weights = attention_mask.unsqueeze(-1)
        pooled = (hidden.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.verifier_head(pooled.to(self.verifier_head.weight.dtype)).float()


class FiveClassRunner(Seq2Seq):
    verifier_loss = VERIFIER_LOSS

    def load_model(self, checkpoint, **kwargs):
        model, info = T5WithStepClassifier.from_pretrained(checkpoint, output_loading_info=True, **kwargs)
        config = model.config
        if getattr(config, "ppnl_verifier_head", None) is None:
            if getattr(config, "ppnl_verifier_loss", None) is not None:
                raise ValueError("Expected the pretrained base or a five-class checkpoint")
            if set(info["missing_keys"]) != {"verifier_head.weight", "verifier_head.bias"}:
                raise ValueError(f"Unexpected missing initial weights: {info['missing_keys']}")
            model.verifier_head.reset_parameters()
        elif (config.ppnl_verifier_head != HEAD or tuple(config.ppnl_verifier_labels) != CLASS_LABELS
              or config.ppnl_verifier_loss != VERIFIER_LOSS or info["missing_keys"]):
            raise ValueError("Five-class checkpoint head, label order, loss or weights do not match")
        return model

    def prepare_training(self, dataset):
        if self.policy_loss != "optimal_actions":
            raise ValueError("G requires all optimal directions")
        rows = model_io.read_split(dataset, "train", version=g_probes.VERSION, fingerprint=g_probes.template_hash())
        self.model.config.ppnl_prompt_sha256 = g_probes.template_hash()
        self.model.config.ppnl_verifier_loss = VERIFIER_LOSS
        self.model.config.ppnl_verifier_head = HEAD
        self.model.config.ppnl_verifier_labels = list(CLASS_LABELS)
        return self.encode_rows(rows)

    def encode_rows(self, rows):
        tokens = self.tokenizer([row["input"] for row in rows], truncation=False)
        features = []
        for i, row in enumerate(rows):
            item = {key: values[i] for key, values in tokens.items()}
            if len(item["input_ids"]) > self.input_limit:
                raise ValueError(f"Training input exceeds configured limit: {row['id']}")
            allowed = [name in row.get("optimal_actions", ()) for name in NAMES]
            if row["mode"] == "v_step" and not any(allowed):
                item["labels"] = CLASS_LABELS.index(row["target"])
            elif row["mode"] == "g_next" and any(allowed) and row["target"] in row["optimal_actions"]:
                item["labels"] = -100
            else:
                raise ValueError(f"Invalid task or action-set feature: {row['id']}")
            features.append(dict(item, policy_actions=allowed))
        return features

    def collate(self, rows):
        encodings = [{k: v for k, v in row.items() if k not in ("labels", "policy_actions")} for row in rows]
        batch = self.tokenizer.pad(encodings, padding=True, pad_to_multiple_of=8, return_tensors="pt")
        batch["labels"] = torch.tensor([row["labels"] for row in rows], dtype=torch.long)
        batch["policy_actions"] = torch.tensor([row["policy_actions"] for row in rows], dtype=torch.bool)
        return batch

    def batch_loss(self, batch):
        return self.loss(batch.to(self.device))

    def loss(self, batch, *, split_verification_loss=False):
        if split_verification_loss:
            raise ValueError("Split token loss is incompatible with the five-class verifier")
        mask, allowed, labels = batch["attention_mask"], batch["policy_actions"], batch["labels"]
        hidden = self.model.get_encoder()(input_ids=batch["input_ids"], attention_mask=mask).last_hidden_state
        is_g = allowed.any(dim=1)
        total = hidden.new_zeros((), dtype=torch.float32)
        if is_g.any():
            g_hidden, g_mask = hidden[is_g], mask[is_g]
            start = g_mask.new_full((len(g_mask), 1), self.model.config.decoder_start_token_id)
            output = self.model(encoder_outputs=BaseModelOutput(last_hidden_state=g_hidden),
                                attention_mask=g_mask, decoder_input_ids=start, use_cache=False)
            total = optimal_action_loss(output.logits[:, 0, self.action_ids], allowed[is_g]).sum()
        if (~is_g).any():
            v_logits = self.model.classify(hidden[~is_g], mask[~is_g])
            total = total + F.cross_entropy(v_logits, labels[~is_g], reduction="sum")
        return total / len(labels)

    def predict(self, inputs, check=lambda: None):
        self.model.eval()
        results = []
        size = self.config["training"]["eval_batch_size"]
        for offset in range(0, len(inputs), size):
            check()
            batch = self.tokenizer(inputs[offset:offset + size], padding=True, pad_to_multiple_of=8,
                                   truncation=False, return_tensors="pt").to(self.device)
            if batch.input_ids.shape[1] > self.input_limit:
                raise ValueError("Verifier input exceeds configured limit")
            with torch.inference_mode(), self.autocast():
                hidden = self.model.get_encoder()(**batch).last_hidden_state
                indices = self.model.classify(hidden, batch.attention_mask).argmax(dim=-1).tolist()
            results.extend(CLASS_LABELS[i] for i in indices)
        return results
