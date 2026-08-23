"""Train a model per-subject on a BCI dataset (within-subject T->E).

Supports BCI Competition IV 2a (22ch, 4-class) and 2b (3ch, 2-class) via
the --dataset flag. The model registry uses dataset-aware factories so
the same model definition works on both.

Run from repo root:
    python -m src.train_all_subjects --dataset bci2a --model eegnet --epochs 300
    python -m src.train_all_subjects --dataset bci2b --model plv_eegnet --epochs 300
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.datasets.bci2a import BCI2aNPZ, trial_id_offset
from src.models.eegnet import EEGNet
from src.models.phase_eegnet import PhaseEEGNet
from src.models.phase_transformer import PhaseTransformer
from src.models.phase_transformer_ct import PhaseTransformerCT
from src.models.plv_eegnet import PLVEEGNet
from src.models.tiny_transformer import TinyTransformer
from src.train import _pick_device, run_validation, set_seed


# --- dataset registry -----------------------------------------------------

DATASET_INFO = {
    "bci2a": {
        "n_channels": 22,
        "n_classes": 4,
        "n_samples": 1000,
        "sampling_rate": 250,
        "data_dir": "data/bci2a",
        "subject_label_fmt": "A0{sid}",          # A01, A02, ..., A09
        "filename_fmt": "A0{sid}{suffix}.npz",   # e.g. "A01T.npz"
    },
    "bci2b": {
        "n_channels": 3,
        "n_classes": 2,
        "n_samples": 1000,
        "sampling_rate": 250,
        "data_dir": "data/bci2b",
        "subject_label_fmt": "B0{sid}",          # B01, ..., B09
        "filename_fmt": "B0{sid}{suffix}.npz",
    },
    "lee2019": {
        "n_channels": 62,
        "n_classes": 2,
        "n_samples": 1000,
        "sampling_rate": 250,
        "data_dir": "data/lee2019",
        # Two-digit zero-padding: L01..L54.
        "subject_label_fmt": "L{sid:02d}",
        "filename_fmt": "L{sid:02d}{suffix}.npz",
    },
    "schirrmeister": {
        "n_channels": 128,
        "n_classes": 4,
        "n_samples": 1000,
        "sampling_rate": 250,
        "data_dir": "data/schirrmeister",
        "subject_label_fmt": "S{sid:02d}",
        "filename_fmt": "S{sid:02d}{suffix}.npz",
    },
}


# --- model registry -------------------------------------------------------
# Factories take a dataset-info dict so the same model definition works
# across datasets with different (n_channels, n_classes) shapes.

def _eegnet(info):
    return EEGNet(
        n_channels=info["n_channels"], n_classes=info["n_classes"],
        n_samples=info["n_samples"], sampling_rate=info["sampling_rate"],
        temporal_kernel=125, F1=8, D=2, F2=16, dropout=0.5,
    )


def _tiny_transformer(info):
    return TinyTransformer(
        n_channels=info["n_channels"], n_classes=info["n_classes"],
        n_samples=info["n_samples"],
        patch_size=25, d_model=64, n_heads=4, n_layers=3, ff_dim=256,
        dropout=0.5,
    )


def _phase_transformer(info, *, lam_init=0.0, freeze_lam=False):
    return PhaseTransformer(
        n_channels=info["n_channels"], n_classes=info["n_classes"],
        n_samples=info["n_samples"], sampling_rate=info["sampling_rate"],
        patch_size=25, d_model=64, n_heads=4, n_layers=3, ff_dim=256,
        dropout=0.5, phase_band=(8.0, 30.0),
        lam_init=lam_init, freeze_lam=freeze_lam,
    )


def _phase_transformer_ct(info, *, freeze_lam=False):
    return PhaseTransformerCT(
        n_channels=info["n_channels"], n_classes=info["n_classes"],
        n_samples=info["n_samples"], sampling_rate=info["sampling_rate"],
        patch_size=100, d_model=64, n_heads=4, n_layers=2, ff_dim=128,
        dropout=0.5, phase_band=(8.0, 30.0), lam_init=0.0, freeze_lam=freeze_lam,
    )


def _phase_eegnet(info, *, phase_mode="phase"):
    return PhaseEEGNet(
        n_channels=info["n_channels"], n_classes=info["n_classes"],
        n_samples=info["n_samples"], sampling_rate=info["sampling_rate"],
        phase_band=(8.0, 30.0), phase_mode=phase_mode,
        temporal_kernel=125, F1=8, D=2, F2=16, dropout=0.5,
    )


def _plv_eegnet(info, *, plv_mode="real", bottleneck_dim=None):
    phase_band = info.get("phase_band_override", (8.0, 30.0))
    bn = info.get("bottleneck_dim_override", bottleneck_dim)
    salt = info.get("fixed_perm_salt_override", 0)
    return PLVEEGNet(
        n_channels=info["n_channels"], n_classes=info["n_classes"],
        n_samples=info["n_samples"], sampling_rate=info["sampling_rate"],
        phase_band=phase_band, plv_mode=plv_mode,
        bottleneck_dim=bn, fixed_perm_salt=salt,
        temporal_kernel=125, F1=8, D=2, F2=16, dropout=0.5,
    )


MODEL_REGISTRY = {
    "eegnet":               {"factory": _eegnet,                                          "default_lr": 1.0e-3},
    "tiny_transformer":     {"factory": _tiny_transformer,                                "default_lr": 5.0e-4},
    "phase_transformer":    {"factory": _phase_transformer,                               "default_lr": 5.0e-4},
    "phase_transformer_lam0": {"factory": lambda i: _phase_transformer(i, freeze_lam=True), "default_lr": 5.0e-4},
    "phase_transformer_ct": {"factory": _phase_transformer_ct,                            "default_lr": 5.0e-4, "weight_decay": 1.0e-2},
    "phase_transformer_ct_lam0": {"factory": lambda i: _phase_transformer_ct(i, freeze_lam=True), "default_lr": 5.0e-4, "weight_decay": 1.0e-2},
    "phase_eegnet":         {"factory": _phase_eegnet,                                    "default_lr": 1.0e-3},
    "phase_eegnet_shuffle": {"factory": lambda i: _phase_eegnet(i, phase_mode="shuffle"), "default_lr": 1.0e-3},
    "plv_eegnet":           {"factory": _plv_eegnet,                                      "default_lr": 1.0e-3},
    "plv_eegnet_zero":      {"factory": lambda i: _plv_eegnet(i, plv_mode="zero"),        "default_lr": 1.0e-3},
    "plv_eegnet_shuffle_pairs": {"factory": lambda i: _plv_eegnet(i, plv_mode="shuffle_pairs"), "default_lr": 1.0e-3},
    "plv_eegnet_shuffle_pairs_fixed": {"factory": lambda i: _plv_eegnet(i, plv_mode="shuffle_pairs_fixed"), "default_lr": 1.0e-3},
    "plv_eegnet_shuffle_pairs_train_resampled_eval_fixed": {
        "factory": lambda i: _plv_eegnet(
            i, plv_mode="shuffle_pairs_train_resampled_eval_fixed"
        ),
        "default_lr": 1.0e-3,
    },
    "plv_eegnet_pli":          {"factory": lambda i: _plv_eegnet(i, plv_mode="pli"),           "default_lr": 1.0e-3},
    "plv_eegnet_wpli":         {"factory": lambda i: _plv_eegnet(i, plv_mode="wpli"),          "default_lr": 1.0e-3},
}


def _stratified_split(y: np.ndarray, val_frac: float, seed: int):
    """Return (train_idx, val_idx) lists with classes balanced across val.

    Uses a per-class shuffle so the val fraction is approximately equal
    per class. Deterministic given the seed.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)


def train_one_subject(
    subject: int,
    data_dir: Path,
    info: dict,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
    ckpt_dir: Path | None = None,
    verbose: bool = False,
    inner_val_frac: float = 0.2,
) -> dict:
    """Train one subject with proper train / inner-val / eval split.

    The T set (`{sid}T.npz`) is stratified-split into an inner training
    set (1 - inner_val_frac) and an inner validation set
    (inner_val_frac). Early-stopping / best-epoch selection uses ONLY
    the inner-val accuracy. The E set (`{sid}E.npz`) is evaluated each
    epoch as well, but its values are NOT used to choose a model.

    Reported fields:
      - inner_val_best          inner-val accuracy at the chosen epoch
      - inner_val_best_epoch    epoch index that maximized inner-val
      - eval_at_best_val        E-set accuracy at that chosen epoch
                                (PRIMARY honest metric)
      - eval_last               E-set accuracy at the final epoch
      - eval_oracle_best        max E-set accuracy across epochs
                                (kept for sensitivity / backward
                                compat with prior runs)
    """
    set_seed(seed)
    fname_fmt = info["filename_fmt"]
    bandpass = info.get("bandpass_override", (8.0, 30.0))
    notch = info.get("notch_override", None)
    uses_trial_ids = model_name in {
        "plv_eegnet_shuffle_pairs_fixed",
        "plv_eegnet_shuffle_pairs_train_resampled_eval_fixed",
    }
    full_train = BCI2aNPZ(
        data_dir / fname_fmt.format(sid=subject, suffix="T"),
        bandpass=bandpass,
        notch=notch,
        return_trial_id=uses_trial_ids,
        trial_id_offset=trial_id_offset(subject, evaluation=False),
    )
    eval_set = BCI2aNPZ(
        data_dir / fname_fmt.format(sid=subject, suffix="E"),
        bandpass=bandpass,
        notch=notch,
        return_trial_id=uses_trial_ids,
        trial_id_offset=trial_id_offset(subject, evaluation=True),
    )

    # Inner train/val split out of T, stratified by class. The split
    # seed is derived from the run seed so it is reproducible across
    # conditions but varies across seeds.
    y_T = full_train.y.cpu().numpy() if hasattr(full_train.y, "cpu") else np.asarray(full_train.y)
    inner_train_idx, inner_val_idx = _stratified_split(
        y_T, val_frac=inner_val_frac, seed=seed * 1009 + subject,
    )
    inner_train = Subset(full_train, inner_train_idx)
    inner_val = Subset(full_train, inner_val_idx)

    drop_last_train = (len(inner_train) % batch_size == 1)
    train_loader = DataLoader(inner_train, batch_size=batch_size, shuffle=True,
                              drop_last=drop_last_train)
    inner_val_loader = DataLoader(inner_val, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)

    model = MODEL_REGISTRY[model_name]["factory"](info).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    inner_val_best, inner_val_best_epoch = 0.0, 0
    eval_at_best_val, eval_last, eval_oracle_best = 0.0, 0.0, 0.0
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, total = 0.0, 0
        for batch in train_loader:
            x, y = batch[:2]
            trial_ids = batch[2] if len(batch) == 3 else None
            x, y = x.to(device), y.to(device)
            if trial_ids is not None:
                trial_ids = trial_ids.to(device)
            optimizer.zero_grad()
            logits = (
                model(x, trial_ids=trial_ids) if trial_ids is not None else model(x)
            )
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            total += x.size(0)
        train_loss = loss_sum / max(total, 1)
        _, ival_acc = run_validation(model, inner_val_loader, device, criterion)
        _, e_acc = run_validation(model, eval_loader, device, criterion)
        eval_last = e_acc
        if e_acc > eval_oracle_best:
            eval_oracle_best = e_acc
        if ival_acc > inner_val_best:
            inner_val_best = ival_acc
            inner_val_best_epoch = epoch
            eval_at_best_val = e_acc
            if ckpt_dir is not None:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "subject": subject,
                        "epoch": epoch,
                        "inner_val_acc": ival_acc,
                        "eval_at_best_val": e_acc,
                    },
                    ckpt_dir / f"{info['subject_label_fmt'].format(sid=subject)}_best.pt",
                )
        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            label = info["subject_label_fmt"].format(sid=subject)
            print(
                f"    {label} epoch {epoch:03d}/{epochs}  "
                f"train_loss={train_loss:.4f}  ival={ival_acc:.4f}  e={e_acc:.4f}"
            )

    return {
        "subject": subject,
        # PRIMARY honest metric: E-set accuracy at the inner-val-chosen
        # epoch. Replaces the legacy `best_val_acc` field; downstream
        # analysis scripts use this as `best_val_acc` for back-compat.
        "best_val_acc": float(eval_at_best_val),
        "best_epoch": int(inner_val_best_epoch),
        "inner_val_best": float(inner_val_best),
        "eval_at_best_val": float(eval_at_best_val),
        "eval_last": float(eval_last),
        "eval_oracle_best": float(eval_oracle_best),
        "last_val_acc": float(eval_last),  # legacy field name
        "epochs": int(epochs),
        "elapsed_s": round(time.time() - t_start, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=tuple(DATASET_INFO), default="bci2a",
    )
    parser.add_argument(
        "--model", choices=tuple(MODEL_REGISTRY), default="eegnet",
    )
    parser.add_argument("--data-dir", default=None,
                        help="dir with .npz files (default: per-dataset)")
    parser.add_argument("--out", default=None,
                        help="output JSON (default: results/<dataset>_<model>_<epochs>ep.json)")
    parser.add_argument("--ckpt-dir", default=None,
                        help="checkpoint dir (default: results/checkpoints/<dataset>_<model>)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None,
                        help="learning rate (default: per-model recommended)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="weight decay (default: per-model recommended, else 1e-4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subjects", default="1-9")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--bandpass-low", type=float, default=None,
                        help="input bandpass low edge in Hz (default 8.0); pass with --bandpass-high")
    parser.add_argument("--bandpass-high", type=float, default=None,
                        help="input bandpass high edge in Hz (default 30.0)")
    parser.add_argument("--phase-band-low", type=float, default=None,
                        help="PLV phase-extraction band low edge in Hz (default 8.0)")
    parser.add_argument("--phase-band-high", type=float, default=None,
                        help="PLV phase-extraction band high edge in Hz (default 30.0)")
    parser.add_argument("--notch", type=float, default=None,
                        help="line-noise notch frequency in Hz (e.g. 50 or 60); applied before bandpass")
    parser.add_argument("--plv-bottleneck-dim", type=int, default=None,
                        help="if set, apply a learned PLV projection of this "
                             "width before concatenation with the EEGNet trunk")
    args = parser.parse_args()

    info = dict(DATASET_INFO[args.dataset])  # copy so overrides don't leak
    if args.bandpass_low is not None or args.bandpass_high is not None:
        if args.bandpass_low is None or args.bandpass_high is None:
            parser.error("--bandpass-low and --bandpass-high must be set together")
        info["bandpass_override"] = (args.bandpass_low, args.bandpass_high)
    if args.phase_band_low is not None or args.phase_band_high is not None:
        if args.phase_band_low is None or args.phase_band_high is None:
            parser.error("--phase-band-low and --phase-band-high must be set together")
        info["phase_band_override"] = (args.phase_band_low, args.phase_band_high)
    if args.notch is not None:
        info["notch_override"] = args.notch
    if args.plv_bottleneck_dim is not None:
        info["bottleneck_dim_override"] = args.plv_bottleneck_dim
    # Make the fixed-permutation control draw a different permutation family per
    # seed, so repeated seeds test robustness to the permutation draw (not just
    # to weight init). No effect on non-fixed modes.
    info["fixed_perm_salt_override"] = args.seed
    cfg = MODEL_REGISTRY[args.model]
    if args.lr is None:
        args.lr = cfg["default_lr"]
    if args.weight_decay is None:
        args.weight_decay = cfg.get("weight_decay", 1e-4)
    if args.data_dir is None:
        args.data_dir = info["data_dir"]
    # Default output paths now namespace by dataset to avoid collisions.
    if args.out is None:
        args.out = f"results/{args.dataset}_{args.model}_{args.epochs}ep.json"
    if args.ckpt_dir is None:
        args.ckpt_dir = f"results/checkpoints/{args.dataset}_{args.model}"

    if "-" in args.subjects:
        lo, hi = (int(x) for x in args.subjects.split("-"))
        subjects = list(range(lo, hi + 1))
    else:
        subjects = [int(s) for s in args.subjects.split(",") if s.strip()]

    device = _pick_device()
    print(
        f"[train_all] dataset={args.dataset}  model={args.model}  device={device}  "
        f"subjects={subjects}  epochs={args.epochs}  lr={args.lr:.1e}"
    )

    data_dir = Path(args.data_dir)
    results = []
    t_total = time.time()
    for sid in subjects:
        label = info["subject_label_fmt"].format(sid=sid)
        print(f"[train_all] subject {label} ...")
        r = train_one_subject(
            subject=sid, data_dir=data_dir, info=info, model_name=args.model,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay,
            device=device, seed=args.seed,
            ckpt_dir=Path(args.ckpt_dir) if args.ckpt_dir else None,
            verbose=args.verbose,
        )
        print(
            f"    -> best={r['best_val_acc']:.4f} (epoch {r['best_epoch']})  "
            f"last={r['last_val_acc']:.4f}  in {r['elapsed_s']:.1f}s"
        )
        results.append(r)

    best_accs = np.array([r["best_val_acc"] for r in results])
    last_accs = np.array([r["last_val_acc"] for r in results])
    aggregate = {
        "dataset": args.dataset,
        "model": args.model,
        "subjects": subjects,
        "epochs": args.epochs,
        "lr": args.lr,
        "device": str(device),
        "mean_best": float(best_accs.mean()),
        "std_best": float(best_accs.std()),
        "mean_last": float(last_accs.mean()),
        "std_last": float(last_accs.std()),
        "total_elapsed_s": round(time.time() - t_total, 1),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"per_subject": results, "aggregate": aggregate}, indent=2))

    print()
    print(f"=== aggregate dataset={args.dataset}  model={args.model}  "
          f"subjects={len(subjects)}  epochs={args.epochs}  device={device} ===")
    print(f"  mean best val_acc: {aggregate['mean_best']:.4f} ± {aggregate['std_best']:.4f}")
    print(f"  mean last val_acc: {aggregate['mean_last']:.4f} ± {aggregate['std_last']:.4f}")
    print(f"  total time: {aggregate['total_elapsed_s']:.1f}s")
    print(f"  written: {out}")


if __name__ == "__main__":
    main()
