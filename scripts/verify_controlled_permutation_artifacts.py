"""Strict completeness and consistency audit for the schedule experiment."""

from __future__ import annotations

import json
import hashlib
import struct
from pathlib import Path

import numpy as np


RESULTS = Path("results")
CONFIG = {
    "bci2a": {"subjects": range(1, 10), "epochs": 300, "channels": 22},
    "bci2b": {"subjects": range(1, 10), "epochs": 300, "channels": 3},
    "lee2019": {"subjects": range(1, 55), "epochs": 200, "channels": 62},
    "schirrmeister": {"subjects": range(1, 15), "epochs": 200, "channels": 128},
}
SCHEDULES = {
    "resampled": {
        "model": "plv_eegnet_shuffle_pairs_train_resampled_eval_fixed",
        "tag": (
            "plv_eegnet_shuffle_pairs_train_resampled_eval_fixed_"
            "globaltrialid_exactrandperm_dedicatedrng"
        ),
    },
    "fixed": {
        "model": "plv_eegnet_shuffle_pairs_fixed",
        "tag": (
            "plv_eegnet_shuffle_pairs_fixed_"
            "globaltrialid_exactrandperm_dedicatedrng"
        ),
    },
}
SEEDS = (42, 43, 44)


def _assert_accuracy(value: float, context: str) -> None:
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise AssertionError(f"invalid accuracy {value!r}: {context}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed_assignment_key_sha256(
    *, dataset: str, subject: int, seed: int, n_trials: int, n_pairs: int
) -> str:
    digest = hashlib.sha256(b"fixed-assignment-key-v1\0")
    encoded_dataset = dataset.encode("utf-8")
    digest.update(struct.pack("<Q", len(encoded_dataset)))
    digest.update(encoded_dataset)
    digest.update(struct.pack("<qqqq", subject, seed, n_trials, n_pairs))
    key = str(seed).encode()
    offset = subject * (2 ** 40) + (2 ** 32)
    for index in range(n_trials):
        identifier = offset + index
        identifier_bytes = identifier.to_bytes(8, "little", signed=True)
        permutation_seed = int.from_bytes(
            hashlib.blake2b(identifier_bytes, key=key, digest_size=8).digest(),
            "little",
        ) % (2 ** 31)
        digest.update(struct.pack("<qQ", identifier, permutation_seed))
    return digest.hexdigest()


def _load_training_grid(dataset: str, schedule: str) -> dict[tuple[int, int], float]:
    config = CONFIG[dataset]
    expected_subjects = set(config["subjects"])
    tag = SCHEDULES[schedule]["tag"]
    grid = {}
    for seed in SEEDS:
        path = RESULTS / (
            f"{dataset}_{tag}_seed{seed}_{config['epochs']}ep_propersplit.json"
        )
        payload = json.loads(path.read_text())
        aggregate = payload["aggregate"]
        if aggregate["dataset"] != dataset:
            raise AssertionError(f"dataset metadata mismatch: {path}")
        if aggregate["model"] != SCHEDULES[schedule]["model"]:
            raise AssertionError(f"model metadata mismatch: {path}")
        if int(aggregate["epochs"]) != config["epochs"]:
            raise AssertionError(f"epoch budget mismatch: {path}")
        rows = payload["per_subject"]
        subjects = {int(row["subject"]) for row in rows}
        if subjects != expected_subjects or len(rows) != len(expected_subjects):
            raise AssertionError(f"incomplete or duplicate subject grid: {path}")
        values = []
        for row in rows:
            subject = int(row["subject"])
            value = float(row["eval_at_best_val"])
            _assert_accuracy(value, f"{path} subject={subject}")
            if int(row["epochs"]) != config["epochs"]:
                raise AssertionError(f"subject epoch budget mismatch: {path}")
            if int(row["best_epoch"]) not in range(1, config["epochs"] + 1):
                raise AssertionError(f"invalid selected epoch: {path}")
            if not np.isclose(value, float(row["best_val_acc"]), atol=0, rtol=0):
                raise AssertionError(f"primary metric alias mismatch: {path}")
            grid[(subject, seed)] = value
            values.append(value)
        if not np.isclose(np.mean(values), aggregate["mean_best"], atol=1e-12):
            raise AssertionError(f"aggregate mean does not recompute: {path}")
        if not np.isclose(np.std(values), aggregate["std_best"], atol=1e-12):
            raise AssertionError(f"aggregate SD does not recompute: {path}")
    return grid


def _verify_evaluation(
    dataset: str, schedule: str, training_grid: dict[tuple[int, int], float]
) -> dict[tuple[int, int], str]:
    path = RESULTS / (
        f"{dataset}_{schedule}_globaltrialid_exactrandperm_dedicatedrng_"
        "checkpoint_eval_mc20_reproducible.json"
    )
    payload = json.loads(path.read_text())
    if payload["dataset"] != dataset:
        raise AssertionError(f"evaluation dataset mismatch: {path}")
    if payload["training_model"] != SCHEDULES[schedule]["model"]:
        raise AssertionError(f"evaluation model mismatch: {path}")
    if payload["seeds"] != list(SEEDS) or int(payload["draws"]) != 20:
        raise AssertionError(f"evaluation seed/draw protocol mismatch: {path}")
    if not payload["all_fixed_scores_match_training_records"]:
        raise AssertionError(f"fixed checkpoint reproduction failed: {path}")
    rows = payload["per_subject_seed"]
    indexed = {(int(row["subject"]), int(row["seed"])): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(training_grid):
        raise AssertionError(f"evaluation grid is incomplete or duplicated: {path}")
    for key, row in indexed.items():
        if not row["fixed_matches_recorded"]:
            raise AssertionError(f"fixed score mismatch: {path}, {key}")
        if not np.isclose(row["fixed_accuracy"], training_grid[key], atol=1e-8):
            raise AssertionError(f"training/evaluation score mismatch: {path}, {key}")
        draws = row["resampled_draw_accuracies"]
        if len(draws) != 20:
            raise AssertionError(f"wrong draw count: {path}, {key}")
        for index, value in enumerate(draws):
            _assert_accuracy(float(value), f"{path}, {key}, draw={index}")
        n_channels = CONFIG[dataset]["channels"]
        expected_assignment_hash = _fixed_assignment_key_sha256(
            dataset=dataset,
            subject=key[0],
            seed=key[1],
            n_trials=int(row["n_eval_trials"]),
            n_pairs=n_channels * (n_channels - 1) // 2,
        )
        if row.get("fixed_assignment_key_sha256") != expected_assignment_hash:
            raise AssertionError(f"fixed-assignment key hash mismatch: {path}, {key}")
        if not np.isclose(np.mean(draws), row["resampled_mean_accuracy"], atol=1e-12):
            raise AssertionError(f"draw mean does not recompute: {path}, {key}")
        if not np.isclose(np.std(draws, ddof=1), row["resampled_sd_accuracy"], atol=1e-12):
            raise AssertionError(f"draw SD does not recompute: {path}, {key}")
        artifact_name = row.get("prediction_artifact")
        if artifact_name is None:
            raise AssertionError(f"prediction artifact is missing: {path}, {key}")
        artifact = Path(artifact_name)
        if not artifact.is_file():
            raise AssertionError(f"prediction artifact does not exist: {artifact}")
        if _sha256(artifact) != row.get("prediction_artifact_sha256"):
            raise AssertionError(f"prediction artifact hash mismatch: {artifact}")
        arrays = np.load(artifact)
        probabilities = arrays["resampled_probabilities"]
        fixed_probabilities = arrays["fixed_probabilities"]
        labels = arrays["labels"]
        if probabilities.shape[0] != 20 or probabilities.shape[1] != len(labels):
            raise AssertionError(f"prediction artifact shape mismatch: {artifact}")
        recomputed_draws = (probabilities.argmax(axis=-1) == labels).mean(axis=1)
        # Stored accuracies were produced by a float32 torch.mean; recomputing
        # the same integer counts in NumPy can differ by a few 1e-8.
        if not np.allclose(recomputed_draws, draws, atol=1e-7, rtol=0):
            raise AssertionError(f"draw accuracies do not recompute: {artifact}")
        fixed_accuracy = float((fixed_probabilities.argmax(axis=-1) == labels).mean())
        if not np.isclose(fixed_accuracy, row["fixed_accuracy"], atol=1e-7):
            raise AssertionError(f"fixed accuracy does not recompute: {artifact}")
        ensemble = float((probabilities.mean(axis=0).argmax(axis=-1) == labels).mean())
        if not np.isclose(
            ensemble, row["resampled_probability_ensemble_accuracy"], atol=1e-7
        ):
            raise AssertionError(f"ensemble accuracy does not recompute: {artifact}")
    return {
        key: row["fixed_assignment_key_sha256"] for key, row in indexed.items()
    }


def main() -> None:
    for dataset in CONFIG:
        schedule_hashes = {}
        for schedule in SCHEDULES:
            grid = _load_training_grid(dataset, schedule)
            schedule_hashes[schedule] = _verify_evaluation(dataset, schedule, grid)
            print(
                f"verified dataset={dataset} schedule={schedule} "
                f"subject-seed rows={len(grid)}"
            )
        if schedule_hashes["fixed"] != schedule_hashes["resampled"]:
            raise AssertionError(
                f"fixed assignments differ between training schedules: {dataset}"
            )
        print(f"verified fixed-assignment identity across schedules: {dataset}")
    print("all controlled permutation artifacts verified")


if __name__ == "__main__":
    main()
