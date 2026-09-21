"""Verify and safely unpack the immutable evidence parts."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parent

def main():
    inventory = json.loads((ROOT / "evidence/inventory.json").read_text())
    count = 0
    for part in inventory["parts"]:
        name = part["file"]
        if Path(name).name != name:
            raise ValueError("invalid part path")
        archive = ROOT / "evidence" / name
        if hashlib.sha256(archive.read_bytes()).hexdigest() != part["sha256"]:
            raise ValueError("part checksum: " + name)
        with tarfile.open(archive, "r:gz") as tar:
            seen = set()
            for member in tar:
                path = PurePosixPath(member.name)
                if not member.isfile() or path.is_absolute() or ".." in path.parts or path.parts[0] != "runs":
                    raise ValueError("unsafe member: " + member.name)
                if member.name in seen or member.name not in part["members"]:
                    raise ValueError("unexpected member")
                seen.add(member.name)
                data = tar.extractfile(member).read()
                if hashlib.sha256(data).hexdigest() != part["members"][member.name]:
                    raise ValueError("member checksum: " + member.name)
                target = ROOT.joinpath(*path.parts)
                if not target.resolve().is_relative_to(ROOT):
                    raise ValueError("symlink escape")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.is_symlink() or target.read_bytes() != data:
                        raise ValueError("existing evidence differs: " + member.name)
                else:
                    target.write_bytes(data)
                count += 1
            if seen != set(part["members"]):
                raise ValueError("missing members")
    print(f"Verified and unpacked {count} files from {len(inventory['parts'])} parts.")

if __name__ == "__main__":
    main()
