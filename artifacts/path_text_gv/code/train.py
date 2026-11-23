"""GPU-only Flan training on frozen text. No planning modules are part of the runtime package."""
import argparse
from collections import Counter, defaultdict
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import torch
from transformers import AutoModelForSeq2SeqLM

from evaluation import generate_ids, history_contrast, local_eval, outputs, plan_eval, rollout
from objectives import Codec, Options, compute, epoch_weights, preferences, score
from texts import CLASSES, jsonl, parse, risk, save_json, sha, write_lines


REV = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
BASE_SHA = "1dfb70afdcedceb9f9fae2f9b68e004ad934361fb35b9b2bd50b45ea90790fc8"
SETTINGS = dict(seed=42, epochs_including_common=3, update_cases=32, microbatch=4, lr=0.0001,
                beta=0.1, alpha=0.1, draws=3, proposals_batch=8, gradient_clip=1.0, weight_decay=0.01)


def read(path):
    return json.loads(path.read_bytes())


def report(directory, stage, **kwargs):
    payload = dict(time=time.strftime("%Y-%m-%d %H:%M:%S %z"), stage=stage, **kwargs)
    save_json(directory/"progress.json",payload)
    print(json.dumps(payload,ensure_ascii=True),flush=True)


def checkpoint(model, base, dest):
    pending = dest.with_name(dest.name+".pending")
    pending.mkdir(exist_ok=True)
    model.save_pretrained(pending)
    # T5Config in Transformers 5.20 reinterprets the on-disk flag on every load.
    # Preserve the public Flan checkpoint's independent LM head and unscaled logits.
    config=read(pending/"config.json")
    config["tie_word_embeddings"]=False
    config["scale_decoder_outputs"]=False
    save_json(pending/"config.json",config)
    for name in ("tokenizer.json","tokenizer_config.json","special_tokens_map.json","spiece.model"):
        shutil.copyfile(base/name,pending/name)
    save_json(pending/"weights.json",dict(model_sha256=sha(pending/"model.safetensors")))
    pending.rename(dest)


def model_at(path):
    model = AutoModelForSeq2SeqLM.from_pretrained(str(path),local_files_only=True,dtype=torch.float32).cuda()
    assert not model.config.scale_decoder_outputs
    assert model.encoder.embed_tokens.weight is model.decoder.embed_tokens.weight is model.shared.weight
    assert model.lm_head.weight is not model.shared.weight
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.config.use_cache=False
    return model


def make_options(row,codec,mode,neg,ref,proposals):
    gold = row["answers"]
    if row["task"]=="G":
        return Options([codec.target(a) for a in gold])
    key, target = row["id"], gold[0]
    if mode=="field":
        return Options([codec.target(target)], fields=codec.fields(target))
    if mode=="dpo":
        return Options([codec.target(target),codec.target(neg[key])],reference_gap=ref.get(key,0.0))
    group=proposals[key]
    return Options(group["ids"],risks=group["risks"])


@torch.inference_mode()
def reference_gaps(model,codec,rows,encoded,neg,out,stamp):
    file=out/"dpo_reference.json"
    if file.exists():
        existing=read(file)
        if existing["stamp"]==stamp:
            return existing["gaps"]
    result={}
    indices=[i for i,r in enumerate(rows) if r["task"]=="V"]
    model.eval()
    for j in range(0,len(indices),12):
        ids=indices[j:j+12]
        opt=[make_options(rows[i],codec,"dpo",neg,{},{}) for i in ids]
        with torch.autocast("cuda",dtype=torch.bfloat16):
            sums,_=score(model,[encoded[i] for i in ids],opt)
        p=sums.reshape(-1,2)
        for i,gap in zip(ids,(p[:,0]-p[:,1]).tolist(),strict=True):
            result[rows[i]["id"]]=gap
        if j%1200==0:
            report(out,"frozen_reference",processed=j+len(ids),total=len(indices))
    assert all(math.isfinite(x) for x in result.values())
    save_json(file,dict(stamp=stamp,gaps=result))
    return result


def proposal_bank(model,codec,rows,encoded,out,epoch,draws,batch,stamp):
    file=out/f"mrt_proposals_epoch{epoch}.jsonl"
    receipt=out/f"mrt_proposals_epoch{epoch}.json"
    if receipt.exists() and read(receipt)["stamp"]==stamp and sha(file)==read(receipt)["sha256"]:
        return {x["id"]:x for x in jsonl(file)}
    result={}
    model.eval()
    torch.manual_seed(8181+epoch)
    indices=[i for i,r in enumerate(rows) if r["task"]=="V"]
    for j in range(0,len(indices),batch):
        ids=indices[j:j+batch]
        generated=generate_ids(model,[encoded[i] for i in ids],samples=draws,sample=True)
        for k,i in enumerate(ids):
            r=rows[i]
            gold=r["answers"][0]
            # Keep actual sampled token IDs, including EOS when it was sampled.
            candidates={gold:codec.target(gold)}
            for sequence in generated[k*draws:(k+1)*draws]:
                candidates.setdefault(codec.decode(sequence), sequence)
            result[r["id"]]=dict(id=r["id"],ids=list(candidates.values()),risks=[risk(s,gold) for s in candidates])
        if j%(batch*100)==0:
            report(out,"sampling_MRT",epoch=epoch,processed=j+len(ids),total=len(indices))
    write_lines(file,result.values())
    save_json(receipt,dict(stamp=stamp,sha256=sha(file),draws=draws,unique_candidates=dict(Counter(len(v["ids"]) for v in result.values())),
                           zero_risk_only=sum(not any(v["risks"]) for v in result.values())))
    return result


def groups_for(rows):
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        groups[r["case"]].append(i)
    return list(groups.values())


def smoke(args,codec,rows,enc):
    directory=args.out/"smoke"
    directory.mkdir(exist_ok=True)
    model=model_at(args.base)
    neg,details=preferences(rows,42)
    by_label={k:next(i for i,r in enumerate(rows) if r["label"]==k) for k in CLASSES}
    ids=[next(i for i,r in enumerate(rows) if r["task"]=="G" and len(r["answers"])==3), *by_label.values(),
         next(i for i,r in enumerate(rows) if r["task"]=="V" and (parse(r["answers"][0])[1] or 0)>=10)]
    part=[rows[i] for i in ids]
    encpart=[enc[i] for i in ids]
    ref=reference_gaps(model,codec,part,encpart,neg,directory,"smoke-v1")
    bank=proposal_bank(model,codec,part,encpart,directory,1,3,2,"smoke-v1")
    checks={}
    model.eval()
    before=outputs(model,codec,encpart)
    logp_options=[make_options(r,codec,"field",neg,ref,bank) for r in part]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        pre_save=score(model,encpart,logp_options)[0].cpu()
    for mode in ("field","dpo","mrt"):
        model.zero_grad(set_to_none=True)
        # Test V gradients on their own, so that G cannot hide a detached V loss.
        vpart=part[1:]
        for i in range(0,len(vpart),4):
            chunk=vpart[i:i+4]
            opts=[make_options(r,codec,mode,neg,ref,bank) for r in chunk]
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss=compute(model,encpart[1+i:1+i+len(chunk)],chunk,opts,mode,SETTINGS["beta"],SETTINGS["alpha"]).sum()/len(vpart)
            loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True).item()
        assert norm>0, mode
        checks[mode]=dict(V_gradient_norm=norm)
    model.zero_grad(set_to_none=True)
    opts=[make_options(part[0],codec,"field",neg,ref,bank)]
    with torch.autocast("cuda",dtype=torch.bfloat16):
        g_loss=compute(model,encpart[:1],part[:1],opts,"field",0.1,0.1)[0]
    g_loss.backward()
    checks["G_tie_grad"]=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True).item()
    assert checks["G_tie_grad"]>0
    model.zero_grad(set_to_none=True)
    artifact=directory/"weights"
    if not artifact.exists():
        checkpoint(model,args.base,artifact)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    loaded=model_at(artifact).eval()
    after=outputs(loaded,codec,encpart)
    assert before==after
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        post_save=score(loaded,encpart,logp_options)[0].cpu()
    torch.testing.assert_close(pre_save,post_save,rtol=0,atol=0)
    write_lines(directory/"raw.jsonl",[dict(id=r["id"],text=r["text"],gold=r["answers"],raw=s) for r,s in zip(part,after,strict=True)])
    save_json(directory/"passed.json",dict(checks=checks,device=torch.cuda.get_device_name(),free_decode_reload_identical=True,
                   full_sequence_logp_reload_bitwise_equal=True,
                   dpo_preferences=dict(details),samples=len(after),max_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20))


def run_stage(args,codec,rows,enc):
    mode = "field" if args.stage=="warmup" else args.stage
    directory=args.out/args.stage
    directory.mkdir(exist_ok=True)
    epochs=(1,) if args.stage=="warmup" else (2,3)
    base=args.base if args.stage=="warmup" else args.out/"warmup/epoch1/model"
    weights,masses=epoch_weights(rows)
    groups=groups_for(rows)
    assert len(rows)==20400 and len(groups)==4000
    dev=jsonl(args.data/"dev.jsonl")
    settings=dict(SETTINGS,stage=args.stage,mode=mode,manifest_sha256=sha(args.data/"manifest.json"),
                  device=torch.cuda.get_device_name(),effective_loss_mass=masses,common_epoch1=args.stage!="warmup",
                  base_revision=REV,base_model_sha256=BASE_SHA, initial_weights_sha256=sha(base/"model.safetensors"))
    existing=directory/"settings.json"
    if existing.exists() and read(existing)!=settings:
        raise ValueError("Cannot resume incompatible experiment")
    save_json(existing,settings)
    report(directory,"loading",epochs=epochs)
    model=model_at(base)
    ref,negs={},{}
    if mode=="dpo":
        negs,breakdown=preferences(rows,SETTINGS["seed"])
        ref=reference_gaps(model,codec,rows,enc,negs,directory,settings["initial_weights_sha256"]+settings["manifest_sha256"])
        save_json(directory/"pair_types.json",dict(breakdown))
    optimizer=torch.optim.AdamW(model.parameters(),lr=SETTINGS["lr"],weight_decay=SETTINGS["weight_decay"],fused=True)
    for epoch in epochs:
        ep=directory/f"epoch{epoch}"
        ep.mkdir(exist_ok=True)
        # Atomic optimizer+weights at finished epoch boundaries, also recover a failure during evaluation.
        restart=ep/"optimizer.pt"
        ckpt=ep/"model"
        if ckpt.exists() and restart.exists():
            del model,optimizer
            gc.collect()
            torch.cuda.empty_cache()
            model=model_at(ckpt)
            optimizer=torch.optim.AdamW(model.parameters(),lr=SETTINGS["lr"],weight_decay=SETTINGS["weight_decay"],fused=True)
            optimizer.load_state_dict(torch.load(restart,map_location="cpu",weights_only=True))
        else:
            bank={}
            if mode=="mrt":
                prior=base if epoch==2 else directory/"epoch2/model"
                bank=proposal_bank(model,codec,rows,enc,directory,epoch,SETTINGS["draws"],SETTINGS["proposals_batch"],
                                   read(prior/"weights.json")["model_sha256"]+settings["manifest_sha256"])
            options=[make_options(r,codec,mode,negs,ref,bank) for r in rows]
            rng=random.Random(SETTINGS["seed"]+epoch)
            torch.manual_seed(SETTINGS["seed"]+epoch)
            order=list(groups)
            rng.shuffle(order)
            total=0.0
            visited=bytearray(len(rows))
            model.train()
            report(directory,"training",epoch=epoch,update=0,total_updates=math.ceil(len(groups)/SETTINGS["update_cases"]))
            for start in range(0,len(order),SETTINGS["update_cases"]):
                bunch=order[start:start+SETTINGS["update_cases"]]
                indices=[i for group in bunch for i in group]
                rng.shuffle(indices)
                optimizer.zero_grad(set_to_none=True)
                step_loss=torch.zeros((),device=model.device)
                for j in range(0,len(indices),SETTINGS["microbatch"]):
                    batch=indices[j:j+SETTINGS["microbatch"]]
                    assert not any(visited[i] for i in batch)
                    for i in batch:
                        visited[i]=1
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        losses=compute(model,[enc[i] for i in batch],[rows[i] for i in batch],[options[i] for i in batch],
                                       mode,SETTINGS["beta"],SETTINGS["alpha"])
                        weighted=losses.dot(losses.new_tensor([weights[i] for i in batch]))*len(groups)/(len(rows)*len(bunch))
                    weighted.backward()
                    step_loss+=weighted.detach()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),SETTINGS["gradient_clip"],error_if_nonfinite=True)
                optimizer.step()
                total+=step_loss.item()
                update=start//SETTINGS["update_cases"]+1
                if update in (1,2) or update%10==0 or start+len(bunch)==len(groups):
                    report(directory,"training",epoch=epoch,update=update,total_updates=math.ceil(len(groups)/SETTINGS["update_cases"]),
                           mean_loss=total/update,grad_norm=norm.item(),peak_mib=round(torch.cuda.max_memory_allocated()/2**20))
            assert all(visited)
            optimizer.zero_grad(set_to_none=True)
            pending=ep/"optimizer.pending"
            torch.save(optimizer.state_dict(),pending)
            pending.replace(restart)
            if not ckpt.exists():
                checkpoint(model,args.base,ckpt)
        if not (ep/"metrics.json").exists():
            report(directory,"development",epoch=epoch)
            model.eval()
            metrics=local_eval(model,codec,dev,ep)
            metrics["dev_history_contrast"]=history_contrast(model,codec,args.heldout,ep)
            metrics["full_path"]=plan_eval(model,codec,args.heldout,"dev",ep,["whole_path"])["whole_path"]
            save_json(ep/"metrics.json",metrics)
        report(directory,"epoch_finished",epoch=epoch,V_macro_exact=read(ep/"metrics.json")["V"]["macro_exact"])
    save_json(directory/"finished.json",dict(epochs=epochs,mode=mode))


def main():
    cli=argparse.ArgumentParser()
    for name in ("base","data","heldout","out"):
        cli.add_argument("--"+name,required=True,type=Path)
    cli.add_argument("--stage",choices=("smoke","warmup","field","dpo","mrt","test"),required=True)
    args=cli.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    assert (args.base/"upstream_revision.txt").read_text().strip()==REV
    manifest=read(args.data/"manifest.json")
    assert manifest["version"]=="full-path-text-v1"
    assert sha(args.base/"tokenizer.json")==manifest["tokenizer"]
    assert sha(Path(__file__).with_name("texts.py"))==manifest["prompt_module"]
    assert all(sha(args.data/name)==manifest["files"][name] for name in ("train.jsonl","dev.jsonl","audit.json"))
    codec=Codec(args.base/"tokenizer.json")
    if args.stage=="test":
        final_tests(args,codec)
        return
    if args.stage=="smoke":
        assert sha(args.base/"model.safetensors")==BASE_SHA
    rows=jsonl(args.data/"train.jsonl")
    enc=codec.encode_rows(rows)
    if args.stage=="smoke":
        smoke(args,codec,rows,enc)
    else:
        run_stage(args,codec,rows,enc)


def final_tests(args,codec):
    dest=args.out/"final_eval"
    dest.mkdir(exist_ok=True)
    selected={}
    for mode in ("field","dpo","mrt"):
        candidates=[args.out/"warmup/epoch1",args.out/mode/"epoch2",args.out/mode/"epoch3"]
        def rank(path):
            m=read(path/"metrics.json")
            return m["V"]["macro_exact"],m["full_path"]["binary_macro_f1"],m["G"]["optimal_next"]
        winner=max(candidates,key=rank)
        selected[mode]=dict(epoch_directory=str(winner),development_rank=rank(winner),weights=read(winner/"model/weights.json"))
    save_json(dest/"selection_before_test.json",selected)
    for mode,sel in selected.items():
        folder=dest/mode
        folder.mkdir(exist_ok=True)
        if (folder/"metrics.json").exists():
            continue
        model=model_at(Path(sel["epoch_directory"])/"model").eval()
        result={s:plan_eval(model,codec,args.heldout,s,folder,["whole_path","annotated_prefix"]) for s in ("single","double")}
        result["rollout_dev"]=rollout(model,codec,args.heldout,folder)
        save_json(folder/"metrics.json",result)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    save_json(dest/"finished.json",dict(selected=selected))


if __name__=="__main__":
    main()
