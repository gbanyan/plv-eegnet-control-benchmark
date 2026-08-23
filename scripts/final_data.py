"""Shared loaders for analyses retained in the final DSP manuscript."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
RD = PROJECT / "results"
FD = PROJECT / "figures"
FD.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "BCI 2a": {
        "baseline": "bci2a_eegnet_seed{seed}_300ep_propersplit.json",
        "zero": "bci2a_plv_eegnet_zero_seed{seed}_300ep_propersplit.json",
        "real": "bci2a_plv_eegnet_seed{seed}_300ep_propersplit.json",
    },
    "BCI 2b": {
        "baseline": "bci2b_eegnet_seed{seed}_300ep_propersplit.json",
        "zero": "bci2b_plv_eegnet_zero_seed{seed}_300ep_propersplit.json",
        "real": "bci2b_plv_eegnet_seed{seed}_300ep_propersplit.json",
    },
    "Lee2019": {
        "baseline": "lee2019_eegnet_seed{seed}_200ep_propersplit.json",
        "zero": "lee2019_plv_eegnet_zero_seed{seed}_200ep_propersplit.json",
        "real": "lee2019_plv_eegnet_seed{seed}_200ep_propersplit.json",
    },
    "HGD": {
        "baseline": "schirrmeister_eegnet_seed{seed}_200ep_propersplit.json",
        "zero": "schirrmeister_plv_eegnet_zero_seed{seed}_200ep_propersplit.json",
        "real": "schirrmeister_plv_eegnet_seed{seed}_200ep_propersplit.json",
    },
}


def _load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _per_subject(result: dict) -> dict[int, float]:
    return {row["subject"]: row["best_val_acc"] for row in result["per_subject"]}


def load_dataset(name: str) -> dict:
    """Load baseline, zero-PLV, and real-PLV subject means over seeds 42--44."""
    plan = DATASETS[name]
    output = {"name": name}
    subject_ids = None
    for condition, template in plan.items():
        seed_arrays = []
        for seed in (42, 43, 44):
            values = _per_subject(_load(RD / template.format(seed=seed)))
            if subject_ids is None:
                subject_ids = sorted(values)
            if sorted(values) != subject_ids:
                raise ValueError(f"subject mismatch for {name}/{condition}/seed{seed}")
            seed_arrays.append(np.asarray([values[s] for s in subject_ids], dtype=float))
        output[condition] = np.stack(seed_arrays).mean(axis=0)
    output["sids"] = subject_ids
    return output
