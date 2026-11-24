"""Train three uninterrupted epochs from base Flan and evaluate sealed checkpoints."""
from datetime import datetime
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


def write(path, **fields):
    partial=path.with_suffix(".pending")
    partial.write_text(json.dumps(fields,indent=2)+"\n",encoding="utf-8")
    partial.replace(path)


def status(stage, **fields):
    write(ROOT/"status.json",stage=stage,time=datetime.now().astimezone().isoformat(),**fields)


def run(args, log, cwd):
    with (ROOT/log).open("a",encoding="utf-8") as stream:
        subprocess.run([sys.executable,"-B","-u",*map(str,args)],cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    os.environ.update(PYTHONUTF8="1",PYTHONUNBUFFERED="1",TOKENIZERS_PARALLELISM="false",
                      HF_HUB_OFFLINE="1",HF_HUB_DISABLE_PROGRESS_BARS="1")
    spec=read(ROOT/"spec.json")
    assert spec["epochs"]==spec["fixed_test_epoch"]==3 and spec["optimizer_restarts"]==0
    status("preflight")
    run([ROOT/"preflight.py"],"preflight.log",ROOT)
    assert read(ROOT/"cpu_preflight.json")["verified_V_training_inputs"]==16400
    base=ROOT.parent/spec["base_directory"]
    assert sha(base/"model.safetensors")==spec["base_sha256"]
    if not (TRAIN/"train_result.json").is_file():
        status("training",epochs=3,base_sha256=spec["base_sha256"])
        run(["-m","step_data_v3.train_shared_cases","--data",WORK/"data/step_g_shared_4000",
             "--pretrained",base,"--out",TRAIN,"--epochs",3,
             "--effective-batch",spec["effective_batch_states"],"--microbatch",spec["microbatch_views"]],"training.log",WORK)
    result=read(TRAIN/"train_result.json")
    experiment=result["experiment"]
    assert result["updates"]==375 and experiment["data_manifest"]==spec["training_manifest_sha256"]
    assert experiment["seed"]==spec["seed"] and experiment["learning_rate"]==spec["learning_rate"]
    assert [(d["epoch"],d["states"],d["views"],d["updates"]) for d in result["dev"]]==[(e,4000,20400,e*125) for e in (1,2,3)]
    assert read(TRAIN/"selection.json")["rank"]==["dev_step_macro_f1","dev_step_correct","earlier_epoch"]
    weights={f"epoch{e}":sha(TRAIN/f"checkpoint-epoch-{e}/model.safetensors") for e in (1,2,3)}
    assert len({spec["base_sha256"],*weights.values()})==4
    write(ROOT/"fixed_before_test.json",primary="epoch3",weights=weights,
          dev_selected=result["checkpoint"],test_commit=spec["test_commit"],base=spec["base_sha256"])
    for e in (1,2,3):
        checkpoint=TRAIN/f"checkpoint-epoch-{e}"
        dest=ROOT/"predictions"/f"epoch{e}"
        status("testing",epoch=e,weights=weights[f"epoch{e}"])
        if not (dest/"receipt.json").is_file():
            run(["-m","step_data_v3.evaluate_coordinates","--checkpoint",checkpoint,"--blind",ROOT/"blind","--out",dest],"testing.log",WORK)
        receipt=read(dest/"receipt.json")
        assert receipt["model_sha256"]==weights[f"epoch{e}"] and receipt["template_sha256"]==spec["checkpoint_prompt_sha256"]
        assert receipt["max_input_tokens"]==max(spec["prompt_audit"][k]["max_input_tokens"] for k in ("single","double"))
    with ZipFile(ROOT/"results.pending","w",compression=ZIP_DEFLATED) as bundle:
        names=("train_result.json","train_history.json","selection.json","preflight.json","gpu_ledger.json")
        for file in [ROOT/"spec.json",ROOT/"fixed_before_test.json",ROOT/"cpu_preflight.json",
                     *(TRAIN/n for n in names),*sorted((ROOT/"predictions").rglob("*.json*"))]:
            bundle.write(file,file.relative_to(ROOT))
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
