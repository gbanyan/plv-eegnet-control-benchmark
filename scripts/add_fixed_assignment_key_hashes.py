"""Add auditable fixed-assignment key hashes to existing MC20 JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_controlled_permutation_checkpoints import (
    _fixed_assignment_key_sha256,
)
from src.train_all_subjects import DATASET_INFO


RESULTS = Path("results")


def main() -> None:
    for path in sorted(RESULTS.glob("*_checkpoint_eval_mc20_reproducible.json")):
        payload = json.loads(path.read_text())
        dataset = payload["dataset"]
        channels = int(DATASET_INFO[dataset]["n_channels"])
        n_pairs = channels * (channels - 1) // 2
        for row in payload["per_subject_seed"]:
            row["fixed_assignment_key_sha256"] = _fixed_assignment_key_sha256(
                dataset=dataset,
                subject=int(row["subject"]),
                seed=int(row["seed"]),
                n_trials=int(row["n_eval_trials"]),
                n_pairs=n_pairs,
            )
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"updated: {path}")


if __name__ == "__main__":
    main()
