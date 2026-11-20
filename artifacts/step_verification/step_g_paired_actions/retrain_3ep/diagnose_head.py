"""Read-only checks of logits, target routing, and gradients on the actual GPU checkpoints."""
import argparse
import gc
from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F
from ppnl.data import read_jsonl, write_json, write_jsonl
from state_baselines.run import CONFIG
from step_data_v3 import g_probes
from step_data_v3.five_class import FiveClassRunner
from step_data_v3.schema import State
from step_data_v3.train_gpu import split


# Independent A-E reference from the requested interface.
CLASSES = ["valid", "goal_not_reached", "obstacle_collision", "out_of_bounds", "invalid_action_token"]


@torch.inference_mode()
def scores(runner, rows):
    logits, pooled_norms = [], []
    for offset in range(0, len(rows), 16):
        inp = runner.tokenizer([r["input"] for r in rows[offset:offset+16]], padding=True,
                               pad_to_multiple_of=8, truncation=False, return_tensors="pt").to(runner.device)
        with runner.autocast():
            h = runner.model.get_encoder()(**inp).last_hidden_state
            scores_ = runner.model.classify(h, inp.attention_mask)
        mask = inp.attention_mask.unsqueeze(-1)
        pooled = (h.float()*mask).sum(1) / mask.sum(1)
        pooled_norms.extend(pooled.norm(dim=-1).tolist())
        logits.append(scores_.cpu())
    x = torch.cat(logits)
    y = torch.tensor([CLASSES.index(r["target"]) for r in rows])
    prob = x.softmax(-1)
    top = x.topk(2, dim=-1).values
    # Subtracting the per-input mean removes offsets common to all five scores.
    centered = x - x.mean(1, keepdim=True)
    pred = [CLASSES[i] for i in x.argmax(-1).tolist()]
    summary = {"n":len(rows), "predictions":dict(Counter(pred)), "correct":int((x.argmax(-1)==y).sum()),
               "mean_cross_entropy":F.cross_entropy(x,y).item(), "mean_top_probability":prob.max(-1).values.mean().item(),
               "mean_logits":x.mean(0).tolist(), "std_of_centered_class_logits_across_inputs":centered.std(0).tolist(),
               "top_two_logit_margin_median":(top[:,0]-top[:,1]).median().item(),
               "mean_pooled_hidden_norm":sum(pooled_norms)/len(rows)}
    return summary, x, pred


def objective_checks(runner, rows, states):
    model = runner.model
    chosen = [next(r for r in rows if r["mode"]=="g_next" and len(r["optimal_actions"])>1)]
    chosen.extend(next(r for r in rows if r["mode"]=="v_step" and r["target"]==c) for c in CLASSES)
    features = runner.encode_rows(chosen)
    batch = runner.collate(features).to(runner.device)
    assert batch.labels.tolist() == [-100, 0, 1, 2, 3, 4]
    w, bias = model.verifier_head.weight, model.verifier_head.bias
    enc_weight = model.encoder.final_layer_norm.weight
    dec_weight = model.decoder.final_layer_norm.weight
    # FP32 and dropout off for the analytic gradient comparison. No optimizer steps.
    with torch.no_grad():
        v = runner.collate(features[1:]).to(runner.device)
        h = model.encoder(input_ids=v.input_ids, attention_mask=v.attention_mask).last_hidden_state
        pooled = torch.stack([h[i,mask.bool()].float().mean(0) for i,mask in enumerate(v.attention_mask)])
        logits = pooled @ w.T + bias
        g = runner.collate(features[:1]).to(runner.device)
        start = g.input_ids.new_full((1,1), model.config.decoder_start_token_id)
        action_logits = model(input_ids=g.input_ids, attention_mask=g.attention_mask, decoder_input_ids=start,
                              use_cache=False).logits[0,0,runner.action_ids].float()
        g_loss = -action_logits.softmax(0)[g.policy_actions[0]].sum().log()
        expected = (g_loss + F.cross_entropy(logits, torch.arange(5, device=w.device), reduction="sum")) / 6
    actual = runner.batch_loss(batch)
    torch.testing.assert_close(actual,expected,atol=3e-4,rtol=2e-4)
    del actual
    v_loss = runner.batch_loss(v)
    gw, gb, ge, gd = torch.autograd.grad(v_loss,(w,bias,enc_weight,dec_weight),allow_unused=True)
    delta = (logits.softmax(-1) - torch.eye(5,device=w.device))/5
    torch.testing.assert_close(gb,delta.sum(0),atol=3e-5,rtol=2e-4)
    torch.testing.assert_close(gw,delta.T @ pooled,atol=3e-4,rtol=3e-4)
    assert ge.norm().item()>0 and gd is None
    only_g = runner.batch_loss(g)
    g_encoder, g_head = torch.autograd.grad(only_g,(enc_weight,w),allow_unused=True)
    assert g_head is None
    # Each trial uses the identical real input with a hypothetical class as the only label.
    single = runner.collate(features[1:2]).to(runner.device)
    sample = chosen[1]
    routes = []
    for k, label in enumerate(CLASSES):
        single.labels.fill_(k)
        loss = runner.batch_loss(single)
        grad, = torch.autograd.grad(loss,(bias,))
        assert torch.where(grad<0)[0].tolist()==[k] and torch.all(grad[torch.arange(5,device=w.device)!=k]>0)
        artificial = torch.full((1,5), -1.0, device=w.device)
        artificial[0,k] = 1
        with patch.object(model,"classify",return_value=artificial):
            decoded = g_probes.verify_last(runner,states[sample["state_id"]],sample["candidate"])
        assert decoded==label
        routes.append({"code":"ABCDE"[k],"target_index":k,"decoded_if_highest":decoded,
                       "ce_on_same_input":loss.item(),"bias_gradient":grad.tolist()})
    return {"g_example_id":chosen[0]["id"],"v_example_ids":[r["id"] for r in chosen[1:]],
            "mixed_loss_reference":expected.item(),"mixed_loss_actual":runner.batch_loss(batch).detach().item(),
            "V_head_weight_gradient_norm":gw.norm().item(),"V_encoder_final_norm_gradient_norm":ge.norm().item(),
            "G_encoder_final_norm_gradient_norm":g_encoder.norm().item(),
            "V_decoder_gradient_absent":gd is None,"G_head_gradient_absent":g_head is None,
            "head_gradient_matches_analytic_CE":True,"fixed_input_id_for_routes":sample["id"],"class_routes":routes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    data, out = args.data, args.run
    raw_train, raw_dev = split(data,"train"), split(data,"dev")
    train = [r for r in raw_train if r["mode"]=="v_step"]
    dev = [r for r in raw_dev if r["mode"]=="v_step"]
    states = {r["id"]:State.from_record(r) for r in read_jsonl(data/"states.jsonl")}
    selection = json.loads((out/"selection.json").read_bytes())["epoch"]
    settings = dict(CONFIG["training"],precision="bfloat16" if torch.cuda.is_bf16_supported() else "float32",
                    max_input_tokens=128,max_target_tokens=8,eval_batch_size=16)
    torch.set_num_threads(2)
    cfg = dict(CONFIG,training=settings)
    initial = FiveClassRunner("E:/YanbingChen/phase1_pretrained/flan_t5_base",cfg)
    previous = initial.model.verifier_head.weight.detach().cpu().clone()
    initial_w = previous.clone()
    del initial
    gc.collect()
    torch.cuda.empty_cache()
    report = {"classes":CLASSES,"uniform_cross_entropy":torch.log(torch.tensor(5.)).item(),"epochs":{}}
    for ck in sorted(out.glob("checkpoint-epoch-*")):
        epoch = int(ck.name.rsplit("-",1)[1])
        runner = FiveClassRunner(str(ck),cfg)
        runner.model.eval()
        assert runner.model.config.ppnl_verifier_labels==CLASSES
        weights = runner.model.verifier_head.weight.detach().cpu().clone()
        train_report, train_scores, _ = scores(runner,train)
        dev_report, dev_scores, preds = scores(runner,dev)
        raw = read_jsonl(out/f"train_dev_epoch_{epoch}.jsonl")
        assert [r["id"] for r in raw]==[r["id"] for r in dev] and preds==[r["prediction"] for r in raw]
        write_jsonl(out/f"dev_logits_epoch_{epoch}.jsonl",
                    [dict(r,logits=logits) for r,logits in zip(raw,dev_scores.tolist(),strict=True)])
        y = torch.tensor([CLASSES.index(r["target"]) for r in dev])
        old_precision=runner.precision
        runner.precision="float32"
        _,_,full_precision=scores(runner,dev)
        runner.precision=old_precision
        entry = {"weight_l2_change_from_initial":(weights-initial_w).norm().item(),
                 "weight_l2_change_from_previous_saved":(weights-previous).norm().item(),
                 "weight_shape":list(weights.shape),"bias":runner.model.verifier_head.bias.detach().cpu().tolist(),
                 "train":train_report,"dev":dev_report,"dev_FP32_predictions":dict(Counter(full_precision)),
                 "dev_saved_predictions_reproduced":len(raw),
                 "dev_correct_after_subtracting_train_mean_logits_diagnostic_only":int(((dev_scores-train_scores.mean(0)).argmax(-1)==y).sum())}
        if epoch==1 or epoch==selection:
            entry["loss_and_routes"]=objective_checks(runner,raw_train,states)
        report["epochs"][epoch]=entry
        previous=weights
        write_json(out/"mechanism_check.json",report)
        del runner
        gc.collect()
        torch.cuda.empty_cache()


if __name__=="__main__":
    main()
