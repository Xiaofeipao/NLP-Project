"""Evaluate the unmodified checkpoint with its usual prompt and two example prefixes."""
from datetime import datetime
from hashlib import file_digest
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from zipfile import ZIP_DEFLATED, ZipFile

ROOT=Path(__file__).resolve().parent
ARMS=("baseline","direct","worked")


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


def status(stage, **fields):
    partial=ROOT/"status.pending"
    partial.write_text(json.dumps(dict(stage=stage,time=datetime.now().astimezone().isoformat(),**fields),indent=2)+"\n",encoding="utf-8")
    partial.replace(ROOT/"status.json")


def main():
    os.environ.update(PYTHONUTF8="1",HF_HUB_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
    spec=read(ROOT/"spec.json")
    assert all(sha(ROOT/n)==h for n,h in read(ROOT/"manifest.json").items())
    model=ROOT.parent/spec["checkpoint_directory"]/"training/checkpoint-epoch-3"
    assert sha(model/"model.safetensors")==spec["weights_sha256"]
    for arm in ARMS:
        status("evaluating",arm=arm)
        out=ROOT/"predictions"/arm
        args=[sys.executable,"-B","-u","-m","step_data_v3.evaluate_coordinates",
              "--checkpoint",str(model),"--blind",str(ROOT/"blind"),"--out",str(out)]
        if arm!="baseline":
            args.extend(("--prefix",str(ROOT/(arm+".txt"))))
        with (ROOT/(arm+".log")).open("w",encoding="utf-8") as log:
            subprocess.run(args,cwd=ROOT/"workspace",stdout=log,stderr=subprocess.STDOUT,check=True)
        r=read(out/"receipt.json")
        assert r["model_sha256"]==spec["weights_sha256"] and r["template_sha256"]==spec["checkpoint_prompt_sha256"]
        assert r["prompt_prefix_sha256"]==spec["prefixes"].get(arm) and r["unique_step_inputs"]==2563
    with ZipFile(ROOT/"results.pending","w",compression=ZIP_DEFLATED) as sealed:
        for file in [ROOT/"spec.json",*sorted((ROOT/"predictions").rglob("*.json*"))]:
            sealed.write(file,file.relative_to(ROOT))
    (ROOT/"results.pending").replace(ROOT/"results.zip")
    status("finished",results_sha256=sha(ROOT/"results.zip"))


if __name__=="__main__":
    try:
        main()
    except Exception as err:
        status("failed",error=repr(err),traceback=traceback.format_exc())
        raise
