"""Serialize the approved fresh Flan training and seal blind predictions before scoring."""
import datetime as dt
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import traceback
from zipfile import ZIP_DEFLATED, ZipFile

from preflight import read, sha

ROOT=Path(__file__).resolve().parent
WORK=ROOT/"workspace"
TRAIN=ROOT/"training"


def save(path,obj):
    partial=path.with_suffix(".partial")
    partial.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")
    partial.replace(path)


def status(stage,**extra):
    save(ROOT/"status.json",dict(stage=stage,time=dt.datetime.now().astimezone().isoformat(),**extra))


def run(args,log,cwd):
    with (ROOT/log).open("a",encoding="utf-8") as stream:
        subprocess.run([sys.executable,"-u",*map(str,args)],cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    os.environ.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                      HF_HUB_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    spec=read(ROOT/"spec.json")
    status("preflight")
    run([ROOT/"preflight.py"],"preflight.log",ROOT)
    assert read(ROOT/"cpu_preflight.json")["cpu_preflight"]
    base=ROOT.parent/spec["base_directory"]
    assert sha(base/"model.safetensors")==spec["base_sha256"]
    if not (TRAIN/"train_result.json").is_file():
        status("training",epochs=2,base_sha256=spec["base_sha256"])
        run(["-m","step_data_v3.train_shared_cases","--data",WORK/"data/step_g_shared_4000",
             "--pretrained",base,"--out",TRAIN,
             "--epochs",spec["epochs"],"--effective-batch",spec["effective_batch_states"],
             "--microbatch",spec["microbatch_views"]],"training.log",WORK)
    result=read(TRAIN/"train_result.json")
    assert result["updates"]==250 and result["experiment"]["data_manifest"]==spec["training_manifest_sha256"]
    assert read(TRAIN/"selection.json")["rank"]==["dev_step_macro_f1","dev_step_correct","earlier_epoch"]
    assert [(d["epoch"],d["states"],d["views"]) for d in result["dev"]]==[(1,4000,20400),(2,4000,20400)]
    new=TRAIN/"checkpoint-epoch-2"
    old=ROOT.parent/spec["previous_directory"]
    checkpoint={"new_epoch2":new,"old_epoch2":old/"checkpoint-epoch-2","old_epoch3":old/"checkpoint-epoch-3"}
    weights={n:sha(p/"model.safetensors") for n,p in checkpoint.items()}
    w1=sha(TRAIN/"checkpoint-epoch-1/model.safetensors")
    assert len({spec["base_sha256"],w1,weights["new_epoch2"]})==3
    assert all(weights[n]==h for n,h in spec["original_weights"].items())
    save(ROOT/"fixed_before_test.json",dict(primary="new_epoch2",base=spec["base_sha256"],
          new_epoch1=w1,evaluated=weights,dev_selected=result["checkpoint"],test_commit=spec["test_commit"]))
    for n,p in checkpoint.items():
        dest=ROOT/"predictions"/n
        if not (dest/"receipt.json").is_file():
            status("testing",checkpoint=n,weights=weights[n])
            args=["-m","step_data_v3.evaluate_coordinates","--checkpoint",p,"--blind",ROOT/"blind","--out",dest]
            if n.startswith("old_"):
                args.append("--original")
            run(args,"testing.log",WORK)
        assert read(dest/"receipt.json")["model_sha256"]==weights[n]
    with ZipFile(ROOT/"results.pending","w",compression=ZIP_DEFLATED) as z:
        for p in [ROOT/"spec.json",ROOT/"cpu_preflight.json",ROOT/"fixed_before_test.json",
                  *sorted((ROOT/"predictions").rglob("*.json*")),TRAIN/"train_result.json",TRAIN/"selection.json",
                  TRAIN/"preflight.json",TRAIN/"gpu_ledger.json",TRAIN/"train_history.json"]:
            z.write(p,p.relative_to(ROOT))
    (ROOT/"results.pending").replace(ROOT/"results.zip")
    status("finished",results_sha256=sha(ROOT/"results.zip"))


if __name__=="__main__":
    with (ROOT/"queue.lock").open("a+b") as lock:
        if lock.tell()==0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            main()
        except Exception as exc:
            status("failed",error=repr(exc),traceback=traceback.format_exc())
            raise
