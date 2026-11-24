"""Train and evaluate the added epoch in its own run directory."""
from datetime import datetime
from hashlib import file_digest
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import traceback
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
WORK = ROOT/"workspace"
TRAIN = ROOT/"training"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def status(stage, **fields):
    temp=ROOT/"status.tmp"
    temp.write_text(json.dumps(dict(time=datetime.now().astimezone().isoformat(), stage=stage, **fields), indent=2)+"\n",encoding="utf-8")
    temp.replace(ROOT/"status.json")


def run(module,log,arguments):
    with (ROOT/log).open("a",encoding="utf-8") as stream:
        subprocess.run([sys.executable,"-B","-u","-m",module,*map(str,arguments)],
                       cwd=WORK,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    os.environ.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                      HF_HUB_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    spec=read(ROOT/"spec.json")
    assert all(sha(ROOT/n)==h for n,h in read(ROOT/"manifest.json").items())
    parent=ROOT.parent/spec["parent_directory"]
    assert read(parent/"status.json")["stage"]=="finished" and sha(parent/"results.zip")==spec["parent_results_sha256"]
    start=parent/"training/checkpoint-epoch-2"
    base=ROOT.parent/spec["base_directory"]
    assert sha(start/"model.safetensors")==spec["parent_weights_sha256"]
    assert sha(base/"tokenizer.json")==spec["tokenizer_sha256"]
    assert spec["V_fields"]==["Current","Goal","Walls","Action"] and spec["added_epochs"]==[3]
    status("training",epoch=3,parent_weights=spec["parent_weights_sha256"],optimizer_restart=True)
    run("step_data_v3.train_shared_cases","training.log",[
        "--data",WORK/"data/step_g_shared_4000","--pretrained",base,"--resume",start,
        "--out",TRAIN,"--epochs",3,"--effective-batch",spec["effective_batch_states"],"--microbatch",spec["microbatch_views"],
    ])
    result=read(TRAIN/"train_result.json")
    epoch,=result["dev"]
    assert result["updates"]==spec["expected_updates"] and result["checkpoint"]=="checkpoint-epoch-3"
    assert (epoch["epoch"],epoch["states"],epoch["views"])==(3,4000,20400)
    experiment=result["experiment"]
    assert experiment["data_manifest"]==spec["training_manifest_sha256"]
    assert (experiment["seed"],experiment["learning_rate"],experiment["endpoint_check"])==(spec["seed"],spec["learning_rate"],False)
    checkpoint=TRAIN/result["checkpoint"]
    weights=sha(checkpoint/"model.safetensors")
    assert weights!=spec["parent_weights_sha256"]
    status("testing",epoch=3,model_sha256=weights)
    output=ROOT/"predictions"
    run("step_data_v3.evaluate_coordinates","testing.log",["--checkpoint",checkpoint,"--blind",ROOT/"blind","--out",output])
    receipt=read(output/"receipt.json")
    assert receipt["model_sha256"]==weights and receipt["template_sha256"]==spec["checkpoint_prompt_sha256"]
    with ZipFile(ROOT/"results.pending","w",compression=ZIP_DEFLATED) as bundle:
        names=("train_result.json","preflight.json","dev_steps_epoch_3.jsonl","dev_directions_epoch_3.jsonl")
        for file in [ROOT/"spec.json",*(TRAIN/n for n in names),*(output/n for n in ("receipt.json","single.jsonl","double.jsonl"))]:
            bundle.write(file,file.relative_to(ROOT))
    (ROOT/"results.pending").replace(ROOT/"results.zip")
    status("finished",epoch=3,model_sha256=weights,results_sha256=sha(ROOT/"results.zip"))


if __name__=="__main__":
    with (ROOT/"lock").open("a+b") as lock:
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
