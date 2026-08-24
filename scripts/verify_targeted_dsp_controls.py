"""Fail-closed validation for the targeted DSP-control result files."""
from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEEDS = (42, 43, 44)
DATASETS = {
    "lee2019": {"epochs": 200, "subjects": list(range(1, 55))},
    "schirrmeister": {"epochs": 200, "subjects": list(range(1, 15))},
}
CONDITIONS = {
    "global_fixed": "plv_eegnet_shuffle_pairs_global_fixed",
    "trial_fixed_no_norm": "plv_eegnet_shuffle_pairs_fixed_no_norm",
    "train_resampled_eval_fixed_no_norm": (
        "plv_eegnet_shuffle_pairs_train_resampled_eval_fixed_no_norm"
    ),
}


def _validate(path: Path, dataset: str, model: str, spec: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    aggregate = payload.get("aggregate", {})
    rows = payload.get("per_subject")
    if aggregate.get("dataset") != dataset:
        raise ValueError(f"dataset mismatch: {path}")
    if aggregate.get("model") != model:
        raise ValueError(f"model mismatch: {path}")
    if aggregate.get("epochs") != spec["epochs"]:
        raise ValueError(f"epoch mismatch: {path}")
    if aggregate.get("subjects") != spec["subjects"]:
        raise ValueError(f"aggregate subject mismatch: {path}")
    if not isinstance(rows, list) or len(rows) != len(spec["subjects"]):
        raise ValueError(f"row-count mismatch: {path}")
    rows = sorted(rows, key=lambda row: int(row["subject"]))
    if [int(row["subject"]) for row in rows] != spec["subjects"]:
        raise ValueError(f"per-subject mismatch: {path}")
    for row in rows:
        for key in ("inner_val_best", "eval_at_best_val", "eval_last"):
            value = float(row[key])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid {key}={value} in {path}")
        best_epoch = int(row["best_epoch"])
        if not 1 <= best_epoch <= spec["epochs"]:
            raise ValueError(f"invalid best_epoch={best_epoch} in {path}")


def main() -> None:
    count = 0
    for dataset, spec in DATASETS.items():
        for tag, model in CONDITIONS.items():
            for seed in SEEDS:
                path = RESULTS / (
                    f"{dataset}_{tag}_seed{seed}_{spec['epochs']}ep_propersplit.json"
                )
                _validate(path, dataset, model, spec)
                count += 1
    print(f"verified {count} targeted DSP-control result files")


if __name__ == "__main__":
    main()
