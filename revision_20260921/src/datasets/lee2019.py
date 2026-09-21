"""Lee2019 (OpenBMI) motor imagery dataset loader.

2-class motor imagery (left hand vs right hand), 54 subjects, 62 EEG
channels, 1000 Hz native (resampled to 250 Hz to match the rest of the
project's pipeline). Loaded via MOABB (Lee2019_MI) from the OpenBMI
mirror.

Note: MOABB's high-level `paradigm.get_data` filters Lee2019 down to a
single session, so we bypass it and load via `_get_single_subject_data`,
which returns both sessions with both `1train` (online calibration) and
`4test` (online testing) runs per session — 4 runs total, ~400 trials
per subject. We then epoch and split:

    train: session 0 (`1train` + `4test`)  — ~200 trials
    eval:  session 1 (`1train` + `4test`)  — ~200 trials

This matches the canonical cross-session protocol used in Lee et al.
2019 (train on session 1, test on session 2).

Output is .npz files with shape compatible with the existing BCI2aNPZ
Dataset class:
    X: float32 [trials, 62, 1000]
    y: int64   [trials]    (0 = left hand, 1 = right hand)

Files are named L01T.npz, L01E.npz, ..., L54T.npz, L54E.npz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Reuse the canonical sampling-rate constant from bci2a.
from src.datasets.bci2a import EXPECTED_SFREQ

# Same label mapping as BCI 2b — left = 0, right = 1.
LEE2019_LABEL_MAP = {"left_hand": 0, "right_hand": 1}
EXPECTED_LEE2019_CHANNELS = 62
# Lee2019 stim channel codes per the dataset's event_id:
# {'left_hand': 2, 'right_hand': 1}.
LEE2019_EVENT_CODE_TO_LABEL = {2: 0, 1: 1}  # 2=left=0, 1=right=1


def _epoch_run(raw, tmin: float, tmax: float, sfreq_target: int):
    """Epoch one Raw run cue-locked. Returns (X, y_int) where X has
    only the 62 EEG channels (EMG + stim dropped) and y_int contains
    integer class labels 0=left, 1=right.

    Order: extract events from stim @ orig sfreq → drop non-EEG →
    resample (rescaling events) → epoch at target sfreq.
    """
    import mne
    raw = raw.copy()
    events = mne.find_events(raw, verbose="ERROR")
    if len(events) == 0:
        return None, None

    # Drop EMG and stim channels — we only want the 62 EEG channels.
    drop = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types())
            if t in ("emg", "stim")]
    if drop:
        raw.drop_channels(drop)

    if int(round(raw.info["sfreq"])) != sfreq_target:
        raw, events = raw.resample(
            sfreq_target, events=events, verbose="ERROR",
        )

    keep_codes = {str(c): c for c in LEE2019_EVENT_CODE_TO_LABEL.keys()
                  if c in events[:, 2]}
    if not keep_codes:
        return None, None
    epochs = mne.Epochs(
        raw, events, event_id=keep_codes, tmin=tmin, tmax=tmax,
        baseline=None, preload=True, verbose="ERROR",
        on_missing="warn",
    )
    X = epochs.get_data()
    y_int = np.array(
        [LEE2019_EVENT_CODE_TO_LABEL[c] for c in epochs.events[:, 2]],
        dtype=np.int64,
    )
    return X, y_int


def load_lee2019_via_moabb(
    subjects,
    out_dir: str | Path,
    tmin: float = 0.0,
    tmax: float = 4.0,
) -> dict[int, dict]:
    """Download Lee2019_MI via MOABB and write per-session .npz files.

    Bypasses `paradigm.get_data` (which filters Lee2019 to a single
    session) by calling `_get_single_subject_data` directly to get all
    4 runs per subject (2 sessions × 2 runs), then epoching each run.

    Args:
        subjects: iterable of 1-based subject ids in 1..54.
        out_dir: output directory for .npz files.
        tmin, tmax: epoch window relative to cue onset (seconds).

    Returns:
        Dict mapping subject id to summary stats.
    """
    try:
        from moabb.datasets import Lee2019_MI
    except ImportError as e:
        raise ImportError(
            "moabb is required for load_lee2019_via_moabb. "
            "Install with `pip install moabb`."
        ) from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = Lee2019_MI(test_run=True)
    target_samples = int(round((tmax - tmin) * EXPECTED_SFREQ))

    summary: dict[int, dict] = {}
    for sid in subjects:
        # Bypass paradigm — call low-level loader directly.
        sessions = dataset._get_single_subject_data(sid)
        # sessions is {session_name: {run_name: Raw}}, with sessions
        # named "0" and "1" and runs "1train" and "4test".

        train_X_parts, train_y_parts = [], []
        eval_X_parts,  eval_y_parts  = [], []
        for sess_name, runs in sessions.items():
            # Concatenate both runs (1train + 4test) within each session.
            for run_name, raw in runs.items():
                X, y = _epoch_run(raw, tmin, tmax, EXPECTED_SFREQ)
                if X is None:
                    continue
                if X.shape[1] != EXPECTED_LEE2019_CHANNELS:
                    raise ValueError(
                        f"subject {sid} session {sess_name} run {run_name}: "
                        f"expected {EXPECTED_LEE2019_CHANNELS} channels, "
                        f"got {X.shape[1]}"
                    )
                if X.shape[-1] > target_samples:
                    X = X[..., :target_samples]
                # Session "0" -> train, session "1" -> eval (canonical
                # cross-session protocol from the Lee2019 paper).
                if sess_name == "0":
                    train_X_parts.append(X.astype(np.float32))
                    train_y_parts.append(y)
                else:
                    eval_X_parts.append(X.astype(np.float32))
                    eval_y_parts.append(y)

        if not train_X_parts or not eval_X_parts:
            raise RuntimeError(
                f"subject {sid}: missing train or eval session data"
            )

        train_X = np.concatenate(train_X_parts, axis=0)
        eval_X = np.concatenate(eval_X_parts, axis=0)
        train_y = np.concatenate(train_y_parts).astype(np.int64)
        eval_y = np.concatenate(eval_y_parts).astype(np.int64)

        train_path = out_dir / f"L{sid:02d}T.npz"
        eval_path = out_dir / f"L{sid:02d}E.npz"
        np.savez_compressed(train_path, X=train_X, y=train_y)
        np.savez_compressed(eval_path, X=eval_X, y=eval_y)

        summary[sid] = {
            "train_trials": int(train_X.shape[0]),
            "eval_trials": int(eval_X.shape[0]),
            "n_channels": int(train_X.shape[1]),
            "n_samples": int(train_X.shape[-1]),
            "train_session": "0",
            "eval_session": "1",
        }
        print(f"  subject L{sid:02d}: {summary[sid]}")

    return summary


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Lee2019_MI to .npz via MOABB.",
    )
    parser.add_argument("--out-dir", required=True, help="output dir for .npz files")
    parser.add_argument(
        "--subjects", default="1-54", help='e.g. "1-54" or "1,3,5"',
    )
    args = parser.parse_args()

    if "-" in args.subjects:
        lo, hi = (int(x) for x in args.subjects.split("-"))
        subjects = list(range(lo, hi + 1))
    else:
        subjects = [int(s) for s in args.subjects.split(",") if s.strip()]

    load_lee2019_via_moabb(subjects, args.out_dir)


if __name__ == "__main__":
    _main()
