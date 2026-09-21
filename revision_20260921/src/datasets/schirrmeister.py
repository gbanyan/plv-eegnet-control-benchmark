"""Schirrmeister 2017 High-Gamma Dataset (HGD) motor-execution loader.

4-class executed movement (left hand, right hand, feet, rest), 14 subjects,
128 EEG channels, 500 Hz native (resampled to 250 Hz to match the rest
of the project's pipeline). Loaded via MOABB (Schirrmeister2017) from
the GIN-hosted EDF mirror.

Each subject has a built-in train/test split: two .edf files at
`data/{train,test}/{sid}.edf`. MOABB exposes them as one session ('0')
with two runs ('0train', '1test'), and the loader maps them to T.npz
(train) and E.npz (test), matching the BCI2a/2b/Lee2019 convention.

Events live in MNE annotations (NOT in a stim channel — opposite of
Lee2019). `mne.find_events` will fail; use `mne.events_from_annotations`.
The numeric event codes returned by that call are assigned in
**alphabetical order** of the annotation strings, so we map labels by
string ('left_hand', 'right_hand', 'feet', 'rest') rather than relying
on numeric codes or `dataset.event_id`.

Output is .npz files with shape compatible with the existing BCI2aNPZ
Dataset class:
    X: float32 [trials, 128, 1000]
    y: int64   [trials]    (0 = left, 1 = right, 2 = feet, 3 = rest)

Files are named S01T.npz, S01E.npz, ..., S14T.npz, S14E.npz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.datasets.bci2a import EXPECTED_SFREQ

EXPECTED_SCHIRR_CHANNELS = 128

SCHIRR_LABEL_MAP = {
    "left_hand": 0,
    "right_hand": 1,
    "feet": 2,
    "rest": 3,
}


def _epoch_run(raw, tmin: float, tmax: float, sfreq_target: int):
    """Epoch one Raw run cue-locked. Returns (X, y_int).

    Order: derive events from annotations @ orig sfreq -> resample
    (rescaling event indices) -> epoch at target sfreq.
    """
    import mne
    raw = raw.copy()

    events, ann_event_id = mne.events_from_annotations(raw, verbose="ERROR")
    if len(events) == 0:
        return None, None

    # ann_event_id maps annotation string -> integer code (alphabetical
    # by string). Invert it so we can recover the string from a code.
    code_to_string = {int(code): str(name) for name, code in ann_event_id.items()}

    # Drop any non-EEG channels just in case (Schirrmeister files are
    # 128-EEG only per the probe, but stay defensive).
    drop = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types())
            if t not in ("eeg",)]
    if drop:
        raw.drop_channels(drop)

    if int(round(raw.info["sfreq"])) != sfreq_target:
        raw, events = raw.resample(
            sfreq_target, events=events, verbose="ERROR",
        )

    # Filter events to ones we know how to label.
    keep_codes = {}
    for code, name in code_to_string.items():
        if name in SCHIRR_LABEL_MAP:
            keep_codes[name] = code
    if not keep_codes:
        return None, None

    epochs = mne.Epochs(
        raw, events, event_id=keep_codes, tmin=tmin, tmax=tmax,
        baseline=None, preload=True, verbose="ERROR",
        on_missing="warn",
    )
    X = epochs.get_data()
    # Map each retained event's code back to a string, then to int label.
    code_to_label = {code: SCHIRR_LABEL_MAP[name]
                     for name, code in keep_codes.items()}
    y_int = np.array(
        [code_to_label[c] for c in epochs.events[:, 2]],
        dtype=np.int64,
    )
    return X, y_int


def load_schirrmeister_via_moabb(
    subjects,
    out_dir: str | Path,
    tmin: float = 0.0,
    tmax: float = 4.0,
) -> dict[int, dict]:
    """Download Schirrmeister2017 via MOABB and write per-subject .npz files.

    Args:
        subjects: iterable of 1-based subject ids in 1..14.
        out_dir: output directory for .npz files.
        tmin, tmax: epoch window relative to cue onset (seconds).

    Returns:
        Dict mapping subject id to summary stats.
    """
    try:
        from moabb.datasets import Schirrmeister2017
    except ImportError as e:
        raise ImportError(
            "moabb is required for load_schirrmeister_via_moabb. "
            "Install with `pip install moabb`."
        ) from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = Schirrmeister2017()
    target_samples = int(round((tmax - tmin) * EXPECTED_SFREQ))

    summary: dict[int, dict] = {}
    for sid in subjects:
        sessions = dataset._get_single_subject_data(sid)
        # Expected layout: {'0': {'0train': raw, '1test': raw}}.

        train_X_parts, train_y_parts = [], []
        eval_X_parts, eval_y_parts = [], []
        for sess_name, runs in sessions.items():
            for run_name, raw in runs.items():
                X, y = _epoch_run(raw, tmin, tmax, EXPECTED_SFREQ)
                if X is None:
                    continue
                if X.shape[1] != EXPECTED_SCHIRR_CHANNELS:
                    raise ValueError(
                        f"subject {sid} session {sess_name} run {run_name}: "
                        f"expected {EXPECTED_SCHIRR_CHANNELS} channels, "
                        f"got {X.shape[1]}"
                    )
                if X.shape[-1] > target_samples:
                    X = X[..., :target_samples]
                # MOABB names runs with substring "train" / "test" in
                # them (e.g. "0train", "1test"). Use that as the split.
                if "train" in run_name:
                    train_X_parts.append(X.astype(np.float32))
                    train_y_parts.append(y)
                elif "test" in run_name:
                    eval_X_parts.append(X.astype(np.float32))
                    eval_y_parts.append(y)
                else:
                    raise ValueError(
                        f"subject {sid}: unexpected run name {run_name!r}"
                    )

        if not train_X_parts or not eval_X_parts:
            raise RuntimeError(
                f"subject {sid}: missing train or eval run data"
            )

        train_X = np.concatenate(train_X_parts, axis=0)
        eval_X = np.concatenate(eval_X_parts, axis=0)
        train_y = np.concatenate(train_y_parts).astype(np.int64)
        eval_y = np.concatenate(eval_y_parts).astype(np.int64)

        train_path = out_dir / f"S{sid:02d}T.npz"
        eval_path = out_dir / f"S{sid:02d}E.npz"
        np.savez_compressed(train_path, X=train_X, y=train_y)
        np.savez_compressed(eval_path, X=eval_X, y=eval_y)

        summary[sid] = {
            "train_trials": int(train_X.shape[0]),
            "eval_trials": int(eval_X.shape[0]),
            "n_channels": int(train_X.shape[1]),
            "n_samples": int(train_X.shape[-1]),
            "train_class_counts": np.bincount(train_y, minlength=4).tolist(),
            "eval_class_counts": np.bincount(eval_y, minlength=4).tolist(),
        }
        print(f"  subject S{sid:02d}: {summary[sid]}")

    return summary


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Schirrmeister2017 HGD to .npz via MOABB.",
    )
    parser.add_argument("--out-dir", required=True, help="output dir for .npz files")
    parser.add_argument(
        "--subjects", default="1-14", help='e.g. "1-14" or "1,3,5"',
    )
    args = parser.parse_args()

    if "-" in args.subjects:
        lo, hi = (int(x) for x in args.subjects.split("-"))
        subjects = list(range(lo, hi + 1))
    else:
        subjects = [int(s) for s in args.subjects.split(",") if s.strip()]

    load_schirrmeister_via_moabb(subjects, args.out_dir)


if __name__ == "__main__":
    _main()
