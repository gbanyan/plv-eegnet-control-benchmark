"""Pack an already curated revision runs/ tree into portable hashed parts."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    files = sorted(p for p in (root / "runs").rglob("*") if p.is_file() and p.suffix in [".json", ".npz"])
    groups, group, size = [], [], 0
    for path in files:
        if group and size + path.stat().st_size > 8_000_000:
            groups.append(group); group = []; size = 0
        group.append(path); size += path.stat().st_size
    if group:
        groups.append(group)
    destination = root / "evidence"
    destination.mkdir(exist_ok=True)
    inventory = {"format": 1, "parts": [], "scope": "training/evaluation/prediction and bank-QA records; no raw EEG, model weights or full surrogate-bank arrays"}
    for index, group in enumerate(groups, 1):
        output = destination / f"records-{index:03d}.tar.gz"
        members = {}
        with output.open("wb") as stream:
            with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w") as tar:
                    for path in group:
                        data = path.read_bytes(); name = path.relative_to(root).as_posix()
                        member = tarfile.TarInfo(name); member.size = len(data); member.mode = 0o644
                        tar.addfile(member, io.BytesIO(data))
                        members[name] = hashlib.sha256(data).hexdigest()
        inventory["parts"].append({"file": output.name, "bytes": output.stat().st_size,
                                   "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "members": members})
    (destination / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps({"files":len(files), "parts":len(groups), "bytes":sum(p["bytes"] for p in inventory["parts"]),
                      "max_part_bytes":max(p["bytes"] for p in inventory["parts"])}))

if __name__ == "__main__":
    main()
