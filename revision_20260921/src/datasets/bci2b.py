"""BCI Competition IV 2b dataset loader.

2-class motor imagery (left hand vs right hand), 9 subjects, 3 EEG channels
(C3, Cz, C4), 250 Hz, ~5 sessions per subject. Loaded via MOABB
(BNCI2014_004) from the BNCI Horizon 2020 mirror.

Standard protocol: first 3 sessions = train, last 2 sessions = eval.
This matches the BCI Competition IV 2b official scoring split.

Output is .npz files with shape compatible with the existing BCI2aNPZ
Dataset class:
    X: float32 [trials, 3, 1000]
    y: int64   [trials]    (0 = left hand, 1 = right hand)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.datasets.bci2a import EXPECTED_SFREQ

# MOABB returns these label strings for BNCI2014_004.
BCI2B_LABEL_MAP = {"left_hand": 0, "right_hand": 1}
EXPECTED_BCI2B_CHANNELS = 3


def load_bci2b_via_moabb(
    subjects,
    out_dir: str | Path,
    tmin: float = 0.0,
    tmax: float = 4.0,
    fmin: float = 0.5,
    fmax: float = 100.0,
) -> dict[int, dict]:
    """Download BCI 2b via MOABB and write per-session .npz files.

    The MOABB MotorImagery paradigm returns all sessions concatenated.
    We split chronologically: first 3 sessions train, last 2 sessions eval,
    which matches the BCI Comp IV 2b official scoring protocol.

    Args:
        subjects: iterable of 1-based subject ids in 1..9.
        out_dir: output directory for .npz files.
        tmin, tmax: epoch window relative to cue onset (seconds).
        fmin, fmax: passband applied by MOABB during epoching. Defaults
            leave the signal effectively unfiltered (BCI2aNPZ does its
            own canonical 8-30 Hz pass at training time).

    Returns:
        Dict mapping subject id to summary stats.
    """
    try:
        from moabb.datasets import BNCI2014_004
        from moabb.paradigms import MotorImagery
    except ImportError as e:
        raise ImportError(
            "moabb is required for load_bci2b_via_moabb. "
            "Install with `pip install moabb`."
        ) from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = BNCI2014_004()
    paradigm = MotorImagery(
        n_classes=2, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, resample=None,
    )
    target_samples = int(round((tmax - tmin) * EXPECTED_SFREQ))

    summary: dict[int, dict] = {}
    for sid in subjects:
        X, y_str, meta = paradigm.get_data(dataset=dataset, subjects=[sid])

        if X.shape[1] != EXPECTED_BCI2B_CHANNELS:
            raise ValueError(
                f"subject {sid}: expected {EXPECTED_BCI2B_CHANNELS} channels "
                f"(C3/Cz/C4), got {X.shape[1]}"
            )

        y = np.array(
            [BCI2B_LABEL_MAP[s] for s in np.asarray(y_str)], dtype=np.int64
        )

        if X.shape[-1] > target_samples:
            X = X[..., :target_samples]
        X = X.astype(np.float32)

        # Split chronologically into train (first 3 sessions) / eval (last 2).
        sessions = sorted(meta["session"].unique())
        if len(sessions) < 2:
            raise RuntimeError(
                f"subject {sid}: expected >= 2 sessions, got {sessions}"
            )
        # The official BCI 2b protocol uses first 3 of 5 sessions as train.
        # If a subject has fewer sessions (rare), fall back to a 60/40 split.
        if len(sessions) >= 5:
            train_sessions = sessions[:3]
            eval_sessions = sessions[3:]
        else:
            split = max(1, int(round(0.6 * len(sessions))))
            train_sessions = sessions[:split]
            eval_sessions = sessions[split:]

        train_mask = meta["session"].isin(train_sessions).to_numpy()
        eval_mask = meta["session"].isin(eval_sessions).to_numpy()

        train_path = out_dir / f"B0{sid}T.npz"
        eval_path = out_dir / f"B0{sid}E.npz"
        np.savez_compressed(train_path, X=X[train_mask], y=y[train_mask])
        np.savez_compressed(eval_path, X=X[eval_mask], y=y[eval_mask])

        summary[sid] = {
            "train_trials": int(train_mask.sum()),
            "eval_trials": int(eval_mask.sum()),
            "n_channels": int(X.shape[1]),
            "n_samples": int(X.shape[-1]),
            "train_sessions": [str(s) for s in train_sessions],
            "eval_sessions": [str(s) for s in eval_sessions],
        }
        print(f"  subject B0{sid}: {summary[sid]}")

    return summary


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert BCI Competition IV 2b to .npz via MOABB.",
    )
    parser.add_argument("--out-dir", required=True, help="output dir for .npz files")
    parser.add_argument(
        "--subjects", default="1-9", help='e.g. "1-9" or "1,3,5"',
    )
    args = parser.parse_args()

    if "-" in args.subjects:
        lo, hi = (int(x) for x in args.subjects.split("-"))
        subjects = list(range(lo, hi + 1))
    else:
        subjects = [int(s) for s in args.subjects.split(",") if s.strip()]

    load_bci2b_via_moabb(subjects, args.out_dir)


if __name__ == "__main__":
    _main()
