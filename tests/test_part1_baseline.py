"""Regression checks for native token alignment, exact first-arm exposure and plan scoring."""

from collections import Counter
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/path_text_gv"))
from gcv_rollout import run_paths
from gcv_schedule import make_streams, schedule_batches
from test_gcv_schedules import reference_responses
from experiments.part1_baseline import policy, protocol
from experiments.part1_baseline.evaluate import evaluate, grade, predictions, structured_verification
from ppnl.core import Grid, OfficialOracle, execute


class Part1PolicyTests(unittest.TestCase):
    def setUp(self):
        self.oracle = OfficialOracle(ROOT / "vendor/ppnl/executor-point-sg.py")
        self.ps = protocol.templates()
        self.grid = dict(rows=6, cols=6, start=[1, 1], goal=[1, 2], obstacles=[[0, 1]])
        self.cases = [
            dict(id=str(i), grid=self.grid, candidate=v)
            for i, v in enumerate(("", "up", "UP", "right", "right left", "right up UP", "right left right"))
        ]

    def test_legacy_plans_and_unchanged_legality(self):
        inputs = self.cases[1:]
        new = policy.run([policy.episode(c, self.ps) for c in inputs], reference_responses)
        self.assertEqual(new, run_paths(inputs, self.ps, reference_responses))
        self.assertEqual(
            [policy.line(r["verification"]) for r in new],
            [execute(Grid.from_dict(self.grid), c["candidate"]).label() for c in inputs],
        )
        self.assertEqual(new[-1]["repair_status"], "unchanged")

    def test_empty_and_paired_end_to_end(self):
        gold = []
        for case in self.cases:
            v = execute(Grid.from_dict(self.grid), case["candidate"])
            gold.append(dict(case, valid=v.valid, label=v.label(), step=v.step, reason=v.reason))
        raw, generated, _ = predictions(self.cases, reference_responses, True, self.oracle)
        self.assertEqual(raw[0]["verification"], dict(label="invalid", step=1, reason="goal_not_reached"))
        self.assertEqual(generated[0]["plan"], "right")
        m, s, e, _ = grade(self.oracle, gold, raw, generated, True)
        self.assertEqual(len(e), 1)
        self.assertEqual(len(s), 7)
        self.assertEqual(m["verification"]["structured_macro_f1"], 1.0)
        self.assertEqual(m["candidate_repair"]["success"], 1.0)
        self.assertTrue(all(p["counts"]["optimal"] == 1 for p in m["pipeline"].values()))

    def test_malformed_counts_as_failure_and_no_oracle_overwrite(self):
        def corrupt(task, texts):
            return ["garbage"] * len(texts) if task == "C" else reference_responses(task, texts)

        c = self.cases[3]
        raw, generated, _ = predictions([c], corrupt, False, self.oracle)
        self.assertEqual(raw[0]["verification"]["label"], "model_error")
        self.assertEqual(generated[0]["status"], "malformed_C")
        d = policy.run(
            [policy.episode(self.cases[1], self.ps)],
            lambda t, s: (["(1,2)"] if t == "C" else ["valid"]) * len(s),
        )
        self.assertEqual(policy.line(d[0]["verification"]), "valid")

    def test_training_budget_and_expanded_regression(self):
        train = protocol.rows_at(ROOT / "data/coordinate_gcv_review/train.jsonl")
        batches, weights, denom = protocol.schedule(train)
        original, prior_weights, _ = make_streams(train)
        self.assertEqual(batches, schedule_batches("uniform", original))
        self.assertLess(max(abs(a - b) for a, b in zip(weights, prior_weights, strict=True)), 1e-12)
        self.assertEqual(denom, 32.0)
        self.assertEqual(Counter(r["task"] for r in train), {"C": 16400, "G": 4000, "V": 16400})
        seen = Counter(i for _, _, batch in batches for i in batch)
        self.assertTrue(all(seen[i] == (1 if r["task"] == "C" else 2) for i, r in enumerate(train)))

    def test_structured_scoring_wrong_step_type_format_and_false_valid(self):
        def invalid(k, t):
            return f"invalid | step={k} | reason={t}"

        wall, bounds = "obstacle_collision", "out_of_bounds"
        rows = [
            {"label": "valid", "reason": None},
            *({"label": invalid(k, wall), "reason": wall} for k in range(1, 5)),
            {"label": invalid(2, bounds), "reason": bounds},
        ]
        predictions = [
            "valid",
            invalid(1, wall),
            invalid(9, wall),
            invalid(3, bounds),
            "valid",
            "invalid|step=2|reason=out_of_bounds",
        ]
        scores = structured_verification(rows, predictions)
        self.assertEqual(
            (scores["exact"], scores["n"], scores["malformed"], scores["false_valid"]), (2, 6, 1, 1)
        )
        self.assertAlmostEqual(scores["exact_accuracy"], 2 / 6)
        self.assertEqual(scores["supported_types"], 3)
        self.assertAlmostEqual(scores["structured_macro_f1"], ((2 / 3) + (2 / 6) + 0) / 3)
        self.assertEqual(scores["per_type"][wall]["fp"], 1)
        self.assertEqual(scores["per_type"][wall]["fn"], 3)
        self.assertIsNone(scores["per_type"]["goal_not_reached"]["f1"])

    def test_hope_dataset_replacement_is_used_in_blind_inputs(self):
        corpus = protocol.ROOT / "data/coordinate_gcv_review"
        source = protocol.read(corpus / "manifest.json")
        replacement = protocol.read(protocol.REPLACEMENT)
        for split, changed in (("single_error", 9), ("double_error", 19)):
            original = protocol.rows_at(corpus / (split.split("_")[0] + "_inputs.jsonl"))
            new, _ = protocol.corrected_error_inputs(
                split, protocol.gold_rows(split), corpus, source, replacement
            )
            self.assertEqual(sum(prev != row for prev, row in zip(original, new, strict=True)), changed)
            self.assertTrue(all(set(row) == {"id", "grid", "candidate"} for row in new))

    def test_original_error_sets_and_loser_cannot_read_test_inputs(self):
        single, double = (protocol.gold_rows(split) for split in protocol.ERROR_SETS)
        self.assertEqual(Counter(r["reason"] for r in single), dict.fromkeys(protocol.LABELS[1:], 50))
        self.assertEqual(
            Counter(r["reason"] for r in double),
            dict(obstacle_collision=68, invalid_action_token=67, out_of_bounds=65),
        )
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            spec = work / "data/protocol.json"
            protocol.put(spec, {"metric": protocol.METRIC})
            protocol.put(
                work / "selection.json",
                dict(
                    protocol=protocol.digest(spec),
                    primary_model="flan_t5_base",
                    scores={"flan_t5_base": {"weights": "one"}},
                ),
            )
            with self.assertRaisesRegex(ValueError, "selected on development"):
                evaluate(None, work, "single_error", "bart_base", "other")


class Part1LossTests(unittest.TestCase):
    def test_family_prefix_targets_scores_and_gradient(self):
        from tokenizers import Tokenizer
        import torch
        from transformers import (
            BartConfig,
            BartForConditionalGeneration,
            T5Config,
            T5ForConditionalGeneration,
        )
        from experiments.part1_baseline.model import Network

        torch.set_num_threads(2)
        for name in protocol.MODELS:
            net = Network.__new__(Network)
            path = ROOT / "outputs/part1_baseline/models/expanded" / name / "tokenizer.json"
            if not path.exists():
                path = ROOT / "artifacts/part1_baseline/models/expanded" / name / "tokenizer.json"
            if not path.exists():
                path = ROOT / ".cache/verification_augmentation_tokenizers" / name / "tokenizer.json"
            if not path.exists():
                self.skipTest("Cached pinned tokenizers needed for exact content-token test")
            net.tok = Tokenizer.from_file(str(path))
            vocab = net.tok.get_vocab_size()
            net.device = torch.device("cpu")
            if name == "bart_base":
                config = BartConfig(
                    vocab_size=vocab,
                    d_model=16,
                    encoder_layers=1,
                    decoder_layers=1,
                    encoder_attention_heads=2,
                    decoder_attention_heads=2,
                    encoder_ffn_dim=32,
                    decoder_ffn_dim=32,
                )
                net.model = BartForConditionalGeneration(config)
                net.bos, net.pad, net.eos, net.prefix = 0, 1, 2, (2, 0)
            else:
                config = T5Config(
                    vocab_size=vocab,
                    d_model=16,
                    d_kv=8,
                    d_ff=32,
                    num_layers=1,
                    num_heads=2,
                    decoder_start_token_id=0,
                )
                net.model = T5ForConditionalGeneration(config)
                net.bos, net.pad, net.eos, net.prefix = None, 0, 1, (0,)
            net.model.eval()
            net.label_ids = [net.target(t)[0] for t in protocol.LABELS]
            self.assertEqual(len(set(net.label_ids)), 5)
            xs = net.encode(["Output only one label."] * 2)
            options = [[net.target("up"), net.target("left")], [(net.label_ids[1],)]]
            terms = net.loss(xs, options, [None, 1])
            if name == "flan_t5_base":
                from coordinate_gv_loss import view_losses

                original = view_losses(net.model, xs, options, [None, 1], net.label_ids)
                torch.testing.assert_close(terms, original)
            scores, targets = net.logits(xs, options)
            ce = torch.nn.functional.cross_entropy(scores.float().transpose(1, 2), targets, reduction="none")
            expected = torch.stack(
                [-torch.logsumexp(-ce[:2].sum(1), 0), -scores[2, 0, net.label_ids].float().log_softmax(0)[1]]
            )
            torch.testing.assert_close(terms, expected)
            source = torch.tensor(xs[1:2])
            direct = net.model(
                input_ids=source, decoder_input_ids=torch.tensor([net.prefix]), use_cache=False
            ).logits
            torch.testing.assert_close(
                scores[2, 0, net.label_ids], direct[0, -1, net.label_ids], atol=2e-5, rtol=2e-5
            )
            terms.sum().backward()
            self.assertTrue(torch.isfinite(net.model.get_input_embeddings().weight.grad).all())


if __name__ == "__main__":
    unittest.main()
