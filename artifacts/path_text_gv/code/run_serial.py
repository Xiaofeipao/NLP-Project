"""Serial desktop job. Each stage is a fresh process and depends on the previous receipt."""
import argparse
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from texts import save_json, sha


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--root",type=Path,required=True)
    cli.add_argument("--base",type=Path,required=True)
    args=cli.parse_args()
    root=args.root.resolve()
    out=root/"runs"
    out.mkdir(exist_ok=True)
    stages=("smoke","warmup","field","dpo","mrt","test")
    lock=(out/"queue.lock").open("a+b")
    lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    state=dict(queue=list(stages),pid=os.getpid(),started=time.time(),finished=[])
    def update(**values):
        state.update(values)
        state["updated"]=time.time()
        save_json(out/"queue_status.json",state)
    active=None
    try:
        package=json.loads((root/"package.json").read_bytes())
        for file,digest in package["sha256"].items():
            assert sha(root/file)==digest,file
        state["package"]=sha(root/"package.json")
        state["dataset_manifest"]=sha(root/"data/manifest.json")
        update(status="verified")
        for active in stages:
            marker=(out/"smoke/passed.json" if active=="smoke" else out/"final_eval/finished.json" if active=="test" else out/active/"finished.json")
            if not marker.exists():
                command=[sys.executable,"-u",str(root/"code/train.py"),"--base",str(args.base),
                         "--data",str(root/"data"),"--heldout",str(root/"heldout"),"--out",str(out),"--stage",active]
                with (out/(active+".log")).open("a",encoding="utf-8") as log:
                    proc=subprocess.Popen(command,cwd=root/"code",stdout=log,stderr=subprocess.STDOUT)
                    update(status="running",stage=active,child_pid=proc.pid)
                    result=proc.wait()
                    if result:
                        raise RuntimeError(f"{active} exited {result}; see {active}.log")
            if not marker.exists():
                raise RuntimeError(f"Missing completion receipt for {active}")
            state["finished"].append(active)
            update(status="between_stages",child_pid=None)
        update(status="completed",stage=None)
    except BaseException:
        update(status="failed",stage=active,child_pid=None,error=traceback.format_exc())
        raise
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
        lock.close()


if __name__=="__main__":
    main()
