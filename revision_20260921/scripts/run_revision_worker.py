"""Persistent sequential GPU queue with separate training/evaluation processes."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from scripts.revision_manifest import load_manifest, manifest_hash
from src.revision import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--worker", choices=("gblinux", "gblab146"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-fraction", type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "worker.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = load_manifest(args.manifest)
    saved = args.output / "manifest.json"
    if saved.exists() and json.loads(saved.read_text()) != manifest:
        raise RuntimeError("output belongs to another run manifest")
    atomic_json(saved, manifest)
    atomic_json(args.output / "worker.json", {"worker": args.worker, "pid": os.getpid(),
                "manifest_sha256": manifest_hash(manifest), "started_unix_time": time.time()})
    errors = []
    for batch in manifest["batches"]:
        if batch["worker"] != args.worker:
            continue
        completion = args.output / "completed_batches" / f"{batch['name']}.json"
        if completion.exists():
            complete = json.loads(completion.read_text())
            if complete["manifest_sha256"] != manifest_hash(manifest):
                raise RuntimeError("completion belongs to another manifest")
            continue
        # Leave room for existing services and checkpoints. Wait rather than
        # killing unrelated processes or pruning shared storage.
        while True:
            gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
                                           "--format=csv,noheader,nounits"], text=True)
            if float(gpu.strip().splitlines()[0]) >= 3500 and shutil.disk_usage(args.output).free >= 10 * 2**30:
                break
            atomic_json(args.output / "progress.json", {"stage": "waiting_for_resources",
                        "batch": batch["name"], "updated_unix_time": time.time()})
            time.sleep(30)
        common = ["--manifest", str(args.manifest), "--batch", batch["name"],
                  "--data-root", str(args.data_root), "--output", str(args.output),
                  "--memory-fraction", str(args.memory_fraction)]
        succeeded = True
        for stage in ("train_revision", "evaluate_revision"):
            code = 1
            for attempt in (1, 2):
                load_manifest(args.manifest)  # immutable source check on every stage/retry
                code = subprocess.call([sys.executable, "-m", f"scripts.{stage}", *common])
                if code == 0:
                    break
                atomic_json(args.output / "progress.json", {"stage": "retry_pending",
                            "batch": batch["name"], "command": stage, "attempt": attempt,
                            "exit_code": code, "updated_unix_time": time.time()})
                if attempt == 1:
                    time.sleep(30)
            if code:
                errors.append({"batch": batch["name"], "stage": stage, "exit_code": code})
                atomic_json(args.output / "failures.json", {"failures": errors})
                succeeded = False
                break
        subprocess.run([sys.executable, "-m", "scripts.summarize_revision", "--manifest",
                        str(args.manifest), "--roots", str(args.output), "--out",
                        str(args.output / "summary")], check=True)
        print(json.dumps({"batch_finished": batch["name"], "success": succeeded}), flush=True)
    atomic_json(args.output / "progress.json", {"stage": "finished" if not errors else "finished_with_failures",
                "failures": errors, "updated_unix_time": time.time()})
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
