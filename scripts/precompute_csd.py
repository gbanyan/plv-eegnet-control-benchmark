"""Pre-compute CSD-transformed npz files for Lee2019 and Schirrmeister.

CSD (Current Source Density via spherical splines, Perrin et al. 1989,
implemented in mne.preprocessing.compute_current_source_density) is a
spatial high-pass filter that reduces volume-conduction by reweighting
each channel by the surface Laplacian under a spherical-head model.

Pipeline (mirrors existing loaders, with CSD inserted before resample):
    MOABB raw  ->  drop non-EEG  ->  set montage  ->
    CSD  ->  resample to 250 Hz  ->  epoch (cue-locked, 4 s)  ->
    per-channel z-score (in standard pipeline)  ->  save .npz

Output:
    data/lee2019_csd/L{01..54}{T,E}.npz
    data/schirrmeister_csd/S{01..14}{T,E}.npz

These npz files are drop-in replacements: train_all_subjects.py with
`--data-dir data/lee2019_csd` will train on CSD signals using the same
PLV / PLI / wPLI / shuffle / zero / baseline conditions.

Run from project root (on gblinux):
    .venv/bin/python -m scripts.precompute_csd --dataset lee2019
    .venv/bin/python -m scripts.precompute_csd --dataset schirrmeister
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


# Lee2019 channel positions: read directly from MOABB raw via the
# original montage stored by the dataset publishers (Korea Univ.). Lee2019
# raw files come with EEG positions; we just need to load them and call
# raw.set_montage(...) won't be required if positions are present.
# Schirrmeister: channel positions come from the original BrainVision
# .vhdr, which mne.io.read_raw_brainvision parses. MOABB returns these
# raws with positions set already.

LEE2019_TRIAL_TMIN = 0.0
LEE2019_TRIAL_TMAX = 4.0  # exclusive of the last sample
SCHIRR_TRIAL_TMIN = 0.0
SCHIRR_TRIAL_TMAX = 4.0
TARGET_SFREQ = 250
BANDPASS = (8.0, 30.0)


def _ensure_montage(raw) -> "mne.io.Raw":
    """Ensure raw has 3D channel positions; otherwise fall back to a
    standard 10-10 montage by name. Returns the raw (possibly modified)."""
    import mne
    # If montage is already set with valid positions, mne returns one.
    montage = raw.get_montage()
    if montage is not None and len(montage.ch_names) >= len(raw.ch_names):
        # Verify positions are not all NaN.
        pos = montage.get_positions()["ch_pos"]
        valid = sum(1 for v in pos.values()
                    if v is not None and not np.any(np.isnan(v)))
        if valid >= len(raw.ch_names):
            return raw
    # Fallback: try standard_1010 by channel name match.
    print(f"  [montage] no valid raw montage; falling back to standard_1010")
    standard = mne.channels.make_standard_montage("standard_1010")
    raw.set_montage(standard, match_case=False, on_missing="warn",
                    verbose="ERROR")
    return raw


def _process_lee2019(out_dir: Path) -> None:
    import mne
    from mne.preprocessing import compute_current_source_density
    from moabb.datasets import Lee2019_MI
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = Lee2019_MI()
    n_subj = 54
    for sid in range(1, n_subj + 1):
        for sess_idx, sess_label in [(0, "T"), (1, "E")]:
            out = out_dir / f"L{sid:02d}{sess_label}.npz"
            if out.exists():
                print(f"  [skip] {out.name}")
                continue
            print(f"  [run]  L{sid:02d}{sess_label}")
            raws = ds._get_single_subject_data(sid)
            session_keys = sorted(raws.keys())
            sess_key = session_keys[sess_idx]
            run_keys = sorted(raws[sess_key].keys())
            run = raws[sess_key][run_keys[0]]
            raw = run.copy().load_data(verbose="ERROR")
            # Drop non-EEG (EMG, stim).
            keep = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types())
                    if t == "eeg"]
            raw.pick(keep)
            # Ensure montage with 3D positions.
            raw = _ensure_montage(raw)
            # Apply CSD.
            raw_csd = compute_current_source_density(raw, verbose="ERROR")
            # Bandpass on CSD signal.
            raw_csd.filter(BANDPASS[0], BANDPASS[1], method="iir",
                           iir_params={"order": 4, "ftype": "butter"},
                           phase="zero", verbose="ERROR")
            # Resample.
            if int(round(raw_csd.info["sfreq"])) != TARGET_SFREQ:
                raw_csd.resample(TARGET_SFREQ, verbose="ERROR")
            # Find events from stim channel (Lee2019 uses stim ch convention).
            try:
                events = mne.find_events(raw_csd, verbose="ERROR")
            except Exception:
                # Some Lee2019 runs have annotations instead.
                events, _ = mne.events_from_annotations(raw_csd, verbose="ERROR")
            # Lee2019 codes: 1=L, 2=R typically. Filter for those.
            keep_codes = [1, 2]
            keep_mask = np.isin(events[:, 2], keep_codes)
            events = events[keep_mask]
            event_id = {"left": 1, "right": 2}
            epochs = mne.Epochs(
                raw_csd, events, event_id=event_id,
                tmin=LEE2019_TRIAL_TMIN,
                tmax=LEE2019_TRIAL_TMAX - 1.0 / TARGET_SFREQ,
                baseline=None, preload=True, verbose="ERROR",
            )
            X = epochs.get_data().astype(np.float32)
            y = (events[:, 2] - 1).astype(np.int64)  # 0 = L, 1 = R
            np.savez_compressed(out, X=X, y=y)
            print(f"        -> {out.name}  X={X.shape}  y={y.shape}")


def _process_schirrmeister(out_dir: Path) -> None:
    import mne
    from mne.preprocessing import compute_current_source_density
    from moabb.datasets import Schirrmeister2017
    SCHIRR_LABEL_MAP = {
        "left_hand": 0, "right_hand": 1, "feet": 2, "rest": 3,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = Schirrmeister2017()
    n_subj = 14
    for sid in range(1, n_subj + 1):
        # Schirrmeister: session '0' has runs 0train, 1test.
        raws_subj = ds._get_single_subject_data(sid)
        # Flatten to a list of (run_label, raw).
        session_keys = sorted(raws_subj.keys())
        sess_key = session_keys[0]
        for run_label, sess_label in [("0train", "T"), ("1test", "E")]:
            out = out_dir / f"S{sid:02d}{sess_label}.npz"
            if out.exists():
                print(f"  [skip] {out.name}")
                continue
            print(f"  [run]  S{sid:02d}{sess_label}")
            run_keys = list(raws_subj[sess_key].keys())
            # Try matching by label first; fall back to index.
            match = [k for k in run_keys if run_label in k]
            if match:
                run = raws_subj[sess_key][match[0]]
            else:
                idx = 0 if run_label.startswith("0") else 1
                run = raws_subj[sess_key][run_keys[idx]]
            raw = run.copy().load_data(verbose="ERROR")
            # Drop non-EEG.
            keep = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types())
                    if t == "eeg"]
            raw.pick(keep)
            raw = _ensure_montage(raw)
            # Apply CSD.
            raw_csd = compute_current_source_density(raw, verbose="ERROR")
            # Bandpass.
            raw_csd.filter(BANDPASS[0], BANDPASS[1], method="iir",
                           iir_params={"order": 4, "ftype": "butter"},
                           phase="zero", verbose="ERROR")
            # Events from annotations (Schirrmeister convention).
            events, ann_event_id = mne.events_from_annotations(
                raw_csd, verbose="ERROR")
            code_to_string = {int(code): str(name)
                              for name, code in ann_event_id.items()}
            keep_codes = {}
            for code, name in code_to_string.items():
                if name in SCHIRR_LABEL_MAP:
                    keep_codes[name] = code
            if not keep_codes:
                print(f"        -> no labeled events, skipping")
                continue
            # Resample BEFORE epoching to match existing loader behavior.
            if int(round(raw_csd.info["sfreq"])) != TARGET_SFREQ:
                raw_csd, events = raw_csd.resample(
                    TARGET_SFREQ, events=events, verbose="ERROR")
            epochs = mne.Epochs(
                raw_csd, events, event_id=keep_codes,
                tmin=SCHIRR_TRIAL_TMIN,
                tmax=SCHIRR_TRIAL_TMAX - 1.0 / TARGET_SFREQ,
                baseline=None, preload=True, verbose="ERROR",
                on_missing="warn",
            )
            X = epochs.get_data().astype(np.float32)
            code_to_label = {code: SCHIRR_LABEL_MAP[name]
                             for name, code in keep_codes.items()}
            y = np.array([code_to_label[int(c)] for c in events[epochs.selection, 2]],
                         dtype=np.int64)
            np.savez_compressed(out, X=X, y=y)
            print(f"        -> {out.name}  X={X.shape}  y={y.shape}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("lee2019", "schirrmeister"),
                   required=True)
    p.add_argument("--out-dir", default=None,
                   help="default: data/<dataset>_csd")
    args = p.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/{args.dataset}_csd")
    print(f"[csd] dataset={args.dataset} out_dir={out_dir}")
    if args.dataset == "lee2019":
        _process_lee2019(out_dir)
    elif args.dataset == "schirrmeister":
        _process_schirrmeister(out_dir)


if __name__ == "__main__":
    main()
