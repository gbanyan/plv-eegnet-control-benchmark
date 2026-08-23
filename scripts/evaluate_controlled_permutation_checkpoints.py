"""Evaluate controlled-schedule checkpoints under fixed and resampled inference.

The training mode ``plv_eegnet_shuffle_pairs_train_resampled_eval_fixed``
selects checkpoints with deterministic validation and test permutations.  This
script verifies that stored test score and then evaluates the same checkpoint
under repeated independently resampled inference.  It reports both the
distribution of single-draw accuracies and the accuracy obtained by averaging
the predictive probabilities across draws.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.bci2a import BCI2aNPZ, trial_id_offset
from src.train_all_subjects import DATASET_INFO, MODEL_REGISTRY, _pick_device


CONTROLLED_MODEL = "plv_eegnet_shuffle_pairs_train_resampled_eval_fixed"
FIXED_MODEL = "plv_eegnet_shuffle_pairs_fixed"
TRAINING_MODELS = (CONTROLLED_MODEL, FIXED_MODEL)


@torch.no_grad()
def _predict_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    for batch in loader:
        x, y = batch[:2]
        trial_ids = batch[2] if len(batch) == 3 else None
        if trial_ids is not None:
            trial_ids = trial_ids.to(device)
        logits = model(x.to(device), trial_ids=trial_ids)
        probabilities.append(logits.softmax(dim=-1).cpu())
        labels.append(y.cpu())
    return torch.cat(probabilities), torch.cat(labels)


def _accuracy(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    return float((probabilities.argmax(dim=-1) == labels).float().mean().item())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed_assignment_key_sha256(
    *, dataset: str, subject: int, seed: int, n_trials: int, n_pairs: int
) -> str:
    """Hash every input key that determines the fixed eval assignments.

    This is deliberately a key-sequence hash rather than a hash of materialized
    ``randperm`` outputs. Together with the recorded PyTorch environment and
    source hash, it cheaply makes schedule-to-schedule assignment identity
    auditable without storing hundreds of millions of permutation indices.
    """
    digest = hashlib.sha256(b"fixed-assignment-key-v1\0")
    encoded_dataset = dataset.encode("utf-8")
    digest.update(struct.pack("<Q", len(encoded_dataset)))
    digest.update(encoded_dataset)
    digest.update(struct.pack("<qqqq", subject, seed, n_trials, n_pairs))
    key = str(seed).encode()
    offset = trial_id_offset(subject, evaluation=True)
    for index in range(n_trials):
        identifier = offset + index
        identifier_bytes = identifier.to_bytes(8, "little", signed=True)
        permutation_seed = int.from_bytes(
            hashlib.blake2b(identifier_bytes, key=key, digest_size=8).digest(),
            "little",
        ) % (2 ** 31)
        digest.update(struct.pack("<qQ", identifier, permutation_seed))
    return digest.hexdigest()


def evaluate_subject(
    *,
    dataset: str,
    subject: int,
    seed: int,
    draws: int,
    batch_size: int,
    checkpoint_root: Path,
    training_model: str,
    device: torch.device,
    predictions_dir: Path | None = None,
) -> dict:
    info = dict(DATASET_INFO[dataset])
    info["fixed_perm_salt_override"] = seed
    data_dir = Path(info["data_dir"])
    eval_set = BCI2aNPZ(
        data_dir / info["filename_fmt"].format(sid=subject, suffix="E"),
        bandpass=info.get("bandpass_override", (8.0, 30.0)),
        notch=info.get("notch_override", None),
        return_trial_id=True,
        trial_id_offset=trial_id_offset(subject, evaluation=True),
    )
    loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)

    model = MODEL_REGISTRY[training_model]["factory"](info).to(device)
    label = info["subject_label_fmt"].format(sid=subject)
    checkpoint_path = checkpoint_root / f"{label}_best.pt"
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    # Controlled mode in eval() uses the deterministic trial-ID-keyed schedule.
    fixed_probabilities, labels = _predict_probabilities(model, loader, device)
    fixed_accuracy = _accuracy(fixed_probabilities, labels)

    # The legacy mode has identical parameters but redraws on every eval forward.
    model.plv_mode = "shuffle_pairs"
    draw_probabilities: list[torch.Tensor] = []
    draw_accuracies: list[float] = []
    for draw in range(draws):
        draw_seed = seed * 1_000_003 + subject * 1_009 + draw
        model.reset_resampling_stream(draw_seed)
        probabilities, draw_labels = _predict_probabilities(model, loader, device)
        if not torch.equal(draw_labels, labels):
            raise RuntimeError("evaluation label order changed between draws")
        draw_probabilities.append(probabilities)
        draw_accuracies.append(_accuracy(probabilities, labels))

    stacked = torch.stack(draw_probabilities)
    ensemble_accuracy = _accuracy(stacked.mean(dim=0), labels)
    result = {
        "subject": subject,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(state["epoch"]),
        "checkpoint_recorded_eval": float(state["eval_at_best_val"]),
        "fixed_accuracy": fixed_accuracy,
        "fixed_matches_recorded": bool(
            np.isclose(fixed_accuracy, state["eval_at_best_val"], atol=1e-8)
        ),
        "resampled_draw_accuracies": draw_accuracies,
        "resampled_mean_accuracy": float(np.mean(draw_accuracies)),
        "resampled_sd_accuracy": float(np.std(draw_accuracies, ddof=1)),
        "resampled_min_accuracy": float(np.min(draw_accuracies)),
        "resampled_max_accuracy": float(np.max(draw_accuracies)),
        "resampled_probability_ensemble_accuracy": ensemble_accuracy,
        "n_eval_trials": int(len(labels)),
        "fixed_assignment_key_sha256": _fixed_assignment_key_sha256(
            dataset=dataset,
            subject=subject,
            seed=seed,
            n_trials=int(len(labels)),
            n_pairs=int(model.n_pairs),
        ),
    }
    if predictions_dir is not None:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        artifact = predictions_dir / (
            f"{dataset}_{training_model}_subject{subject:02d}_seed{seed}_mc{draws}.npz"
        )
        np.savez_compressed(
            artifact,
            fixed_probabilities=fixed_probabilities.numpy().astype(np.float32),
            resampled_probabilities=stacked.numpy().astype(np.float32),
            labels=labels.numpy().astype(np.int64),
        )
        result["prediction_artifact"] = str(artifact)
        result["prediction_artifact_sha256"] = _sha256(artifact)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_INFO))
    parser.add_argument("--subjects", required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-template", required=True)
    parser.add_argument(
        "--training-model", choices=TRAINING_MODELS, default=CONTROLLED_MODEL
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--predictions-dir",
        help="optional directory for compressed per-draw probabilities and labels",
    )
    args = parser.parse_args()

    if args.draws < 2:
        parser.error("--draws must be at least 2 to estimate inference variability")
    if "-" in args.subjects:
        lo, hi = (int(x) for x in args.subjects.split("-"))
        subjects = list(range(lo, hi + 1))
    else:
        subjects = [int(x) for x in args.subjects.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    device = _pick_device()
    predictions_dir = Path(args.predictions_dir) if args.predictions_dir else None
    all_results: list[dict] = []
    for seed in seeds:
        checkpoint_root = Path(args.checkpoint_template.format(seed=seed))
        for subject in subjects:
            print(
                f"[checkpoint-eval] dataset={args.dataset} seed={seed} "
                f"subject={subject} draws={args.draws}",
                flush=True,
            )
            all_results.append(
                evaluate_subject(
                    dataset=args.dataset,
                    subject=subject,
                    seed=seed,
                    draws=args.draws,
                    batch_size=args.batch_size,
                    checkpoint_root=checkpoint_root,
                    training_model=args.training_model,
                    device=device,
                    predictions_dir=predictions_dir,
                )
            )

    mismatches = [r for r in all_results if not r["fixed_matches_recorded"]]
    output = {
        "dataset": args.dataset,
        "training_model": args.training_model,
        "seeds": seeds,
        "draws": args.draws,
        "batch_size": args.batch_size,
        "device": str(device),
        "prediction_artifacts_included": predictions_dir is not None,
        "all_fixed_scores_match_training_records": not mismatches,
        "fixed_score_mismatch_count": len(mismatches),
        "per_subject_seed": all_results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"[checkpoint-eval] written: {out_path}")
    if mismatches:
        raise RuntimeError(
            f"{len(mismatches)} deterministic scores differ from checkpoint records"
        )


if __name__ == "__main__":
    main()
