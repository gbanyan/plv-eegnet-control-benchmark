"""BCI Competition IV 2a dataset.

Two entry points:
    BCI2aNPZ          - PyTorch Dataset over preprocessed .npz files.
    convert_gdf_to_npz - Convert one A0XT.gdf or A0XE.gdf file to .npz.

The conversion follows the standard BCI 2a protocol:
    - 22 EEG channels (3 EOG channels are dropped by channel type)
    - 250 Hz sampling rate
    - Trial window: cue onset to cue + 4 s -> 1000 samples per trial
    - Classes: left hand (0), right hand (1), feet (2), tongue (3)
    - Artifact-flagged trials (event code 1023) are dropped
    - For evaluation files (A0XE.gdf), the cue codes are anonymized as 783
      and true labels live in a companion A0XE.mat file.

Dataset source (registration required):
    https://www.bbci.de/competition/iv/  -> "Data sets 2a: <…> 4-class motor imagery"
    GDFs:       BCICIV_2a_gdf.zip   (A0{1..9}{T,E}.gdf)
    Eval labels: true_labels.zip    (A0{1..9}E.mat with `classlabel` field)

Expected on-disk layout for the CLI:
    data/raw/bci2a/A01T.gdf  A01E.gdf  A01E.mat  ...  A09T.gdf  A09E.gdf  A09E.mat

Run conversion:
    python -m src.datasets.bci2a --gdf-dir data/raw/bci2a --out-dir data/bci2a
"""

# Expected acquisition parameters for BCI 2a — used as validation guards.
EXPECTED_SFREQ = 250
EXPECTED_EEG_CHANNELS = 22
EXPECTED_EOG_CHANNELS = 3

from pathlib import Path

import numpy as np

from src.preprocessing.filters import bandpass_filter, notch_filter, standardize_per_channel

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # converter does not need torch; only BCI2aNPZ does.
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]

# Mapping from BCI 2a annotation strings to class labels.
TRAIN_CUE_CODES = {
    "769": 0,  # left hand
    "770": 1,  # right hand
    "771": 2,  # feet
    "772": 3,  # tongue
}
EVAL_CUE_CODE = "783"      # all eval cues share this code; true label in .mat
ARTIFACT_CODE = "1023"     # trial-level artifact flag
TRIAL_START_CODE = "768"   # trial onset; cue is +2 s after this in BCI 2a


class BCI2aNPZ(Dataset):
    """Dataset wrapper for a preprocessed .npz file.

    Expected keys:
        X: float array of shape [trials, channels, time]
        y: int array of shape [trials]

    Args:
        npz_path: path to the .npz file.
        sampling_rate: sampling rate of the stored signals (Hz).
        bandpass: optional (low, high) tuple in Hz. None disables filtering.
        standardize: whether to apply per-channel standardization.
    """

    def __init__(
        self,
        npz_path: str | Path,
        sampling_rate: int = 250,
        bandpass: tuple[float, float] | None = (8.0, 30.0),
        notch: float | None = None,
        standardize: bool = True,
    ):
        if torch is None:
            raise ImportError(
                "PyTorch is required to use BCI2aNPZ. Install with `pip install torch`."
            )

        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"dataset file not found: {npz_path}")

        data = np.load(npz_path)
        if "X" not in data or "y" not in data:
            raise KeyError(f"{npz_path} must contain 'X' and 'y' arrays")

        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)

        if X.ndim != 3:
            raise ValueError(f"X must be [trials, channels, time], got {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"y shape {y.shape} incompatible with X {X.shape}")

        if notch is not None:
            X = notch_filter(X, sampling_rate, freq=notch)
        if bandpass is not None:
            X = bandpass_filter(X, sampling_rate, low=bandpass[0], high=bandpass[1])
        if standardize:
            X = standardize_per_channel(X)

        self.X = torch.from_numpy(np.ascontiguousarray(X))
        self.y = torch.from_numpy(y)

        self.n_channels = self.X.shape[1]
        self.n_samples = self.X.shape[2]
        self.n_classes = int(self.y.max().item()) + 1 if len(self.y) else 0

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _id_for(code: str, event_id: dict[str, int]) -> int | None:
    """Return the numeric id for an annotation code, tolerating leading zeros."""
    if code in event_id:
        return event_id[code]
    for key, value in event_id.items():
        if key.lstrip("0") == code.lstrip("0"):
            return value
    return None


def convert_gdf_to_npz(
    gdf_path: str | Path,
    out_path: str | Path,
    labels_mat_path: str | Path | None = None,
    tmin: float = 0.0,
    tmax: float = 4.0,
    drop_artifacts: bool = True,
) -> dict:
    """Convert one BCI 2a GDF file to a .npz with X: [trials, 22, T], y: [trials].

    Args:
        gdf_path: path to A0XT.gdf (training) or A0XE.gdf (evaluation).
        out_path: where to write the .npz file.
        labels_mat_path: required for A0XE.gdf — path to the companion
            A0XE.mat that holds `classlabel`. Ignored for training files.
        tmin, tmax: epoch window in seconds, anchored at cue onset.
        drop_artifacts: drop trials whose start was annotated 1023.

    Returns:
        Dict with summary stats: {n_trials, n_dropped, sfreq, n_samples}.
    """
    try:
        import mne
    except ImportError as e:
        raise ImportError(
            "mne is required for GDF conversion. Install with `pip install mne`."
        ) from e

    gdf_path = Path(gdf_path)
    out_path = Path(out_path)
    if not gdf_path.exists():
        raise FileNotFoundError(f"GDF not found: {gdf_path}")

    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose="ERROR")
    sfreq = int(round(raw.info["sfreq"]))
    if sfreq != EXPECTED_SFREQ:
        raise ValueError(
            f"{gdf_path.name}: expected {EXPECTED_SFREQ} Hz sfreq, got {sfreq}"
        )

    # Drop EOG channels by *type* — MNE assigns 'eog' from the GDF header,
    # which is robust to variations in channel naming across MNE versions.
    eog_idx = mne.pick_types(raw.info, eeg=False, eog=True)
    if len(eog_idx) > 0:
        raw.drop_channels([raw.ch_names[i] for i in eog_idx])

    # Fall back to substring drop if the GDF header didn't tag EOG types.
    leftover_eog = [c for c in raw.ch_names if "EOG" in c.upper()]
    if leftover_eog:
        raw.drop_channels(leftover_eog)

    n_eeg = len(mne.pick_types(raw.info, eeg=True))
    if n_eeg != EXPECTED_EEG_CHANNELS:
        raise ValueError(
            f"{gdf_path.name}: expected {EXPECTED_EEG_CHANNELS} EEG channels "
            f"after EOG drop, got {n_eeg} (channels: {raw.ch_names})"
        )

    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")

    # Decide whether this is an evaluation file (anonymized cues only).
    train_ids = {c: _id_for(c, event_id) for c in TRAIN_CUE_CODES}
    train_ids = {c: i for c, i in train_ids.items() if i is not None}
    eval_id = _id_for(EVAL_CUE_CODE, event_id)
    is_eval = (not train_ids) and eval_id is not None

    if is_eval and labels_mat_path is None:
        raise ValueError(
            f"{gdf_path.name} looks like an evaluation file; "
            "labels_mat_path (A0XE.mat) is required to recover labels."
        )

    if is_eval:
        cue_id_map = {EVAL_CUE_CODE: eval_id}
    elif train_ids:
        cue_id_map = train_ids
    else:
        raise RuntimeError(
            f"no recognizable cue codes in {gdf_path.name}; "
            f"saw event_id keys: {sorted(event_id)}"
        )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=cue_id_map,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose="ERROR",
        on_missing="warn",
    )
    cue_events = epochs.events.copy()  # [n_trials, 3] (sample, prev_id, cue_id)

    # Build labels.
    if is_eval:
        from scipy.io import loadmat

        mat = loadmat(str(labels_mat_path))
        if "classlabel" not in mat:
            raise KeyError(
                f"{labels_mat_path} must contain 'classlabel'; got {list(mat)}"
            )
        y = mat["classlabel"].squeeze().astype(np.int64) - 1  # 1..4 -> 0..3
        if y.shape[0] != cue_events.shape[0]:
            raise ValueError(
                f"label count {y.shape[0]} does not match epoch count "
                f"{cue_events.shape[0]} in {gdf_path.name}"
            )
    else:
        id_to_label = {train_ids[c]: TRAIN_CUE_CODES[c] for c in train_ids}
        y = np.array(
            [id_to_label[e] for e in cue_events[:, 2]], dtype=np.int64
        )

    # Drop artifact-flagged trials. In BCI 2a, event 1023 is emitted at the
    # trial-start sample (event 768) of bad trials, which is ~2 s before cue
    # onset. We drop any cue whose corresponding trial-start is artifact-marked.
    n_dropped = 0
    if drop_artifacts:
        artifact_id = _id_for(ARTIFACT_CODE, event_id)
        trial_start_id = _id_for(TRIAL_START_CODE, event_id)
        if artifact_id is not None and trial_start_id is not None:
            artifact_samples = set(events[events[:, 2] == artifact_id, 0])
            trial_start_samples = events[events[:, 2] == trial_start_id, 0]
            keep = np.ones(cue_events.shape[0], dtype=bool)
            window = int(0.1 * sfreq)  # tolerance for matching cue->trial-start
            for i, cue_sample in enumerate(cue_events[:, 0]):
                # Trial-start is the latest 768 event at or before this cue.
                prior = trial_start_samples[trial_start_samples <= cue_sample]
                if prior.size == 0:
                    continue
                start = prior[-1]
                # Any artifact event within ±window of trial start drops the trial.
                if any(abs(s - start) <= window for s in artifact_samples):
                    keep[i] = False
            n_dropped = int((~keep).sum())
            X_data = epochs.get_data()[keep]
            y = y[keep]
        else:
            X_data = epochs.get_data()
    else:
        X_data = epochs.get_data()

    # MNE epochs are inclusive at tmax, yielding `tmax*sfreq + 1` samples.
    # Trim to the canonical 4 s = 1000 samples window.
    target_samples = int(round((tmax - tmin) * sfreq))
    if X_data.shape[-1] > target_samples:
        X_data = X_data[..., :target_samples]

    X_data = X_data.astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X_data, y=y.astype(np.int64))

    return {
        "n_trials": int(X_data.shape[0]),
        "n_dropped": n_dropped,
        "sfreq": sfreq,
        "n_samples": int(X_data.shape[-1]),
        "n_channels": int(X_data.shape[1]),
    }


def convert_subject(
    subject: int,
    gdf_dir: str | Path,
    out_dir: str | Path,
    labels_dir: str | Path | None = None,
) -> None:
    """Convert both A0{subject}T.gdf and A0{subject}E.gdf for one subject."""
    gdf_dir = Path(gdf_dir)
    out_dir = Path(out_dir)
    labels_dir = Path(labels_dir) if labels_dir else gdf_dir

    train_gdf = gdf_dir / f"A0{subject}T.gdf"
    eval_gdf = gdf_dir / f"A0{subject}E.gdf"
    eval_mat = labels_dir / f"A0{subject}E.mat"

    if train_gdf.exists():
        info = convert_gdf_to_npz(train_gdf, out_dir / f"A0{subject}T.npz")
        print(f"  {train_gdf.name}: {info}")
    else:
        print(f"  skip {train_gdf.name} (not found)")

    if eval_gdf.exists():
        if not eval_mat.exists():
            print(f"  skip {eval_gdf.name} ({eval_mat.name} not found)")
        else:
            info = convert_gdf_to_npz(
                eval_gdf, out_dir / f"A0{subject}E.npz", labels_mat_path=eval_mat
            )
            print(f"  {eval_gdf.name}: {info}")
    else:
        print(f"  skip {eval_gdf.name} (not found)")


# MOABB string labels -> our integer class indices.
MOABB_LABEL_MAP = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}


def load_via_moabb(
    subjects,
    out_dir: str | Path,
    tmin: float = 0.0,
    tmax: float = 4.0,
    fmin: float = 0.5,
    fmax: float = 100.0,
) -> dict[int, dict]:
    """Download BCI 2a via MOABB and write per-session .npz files.

    Uses MOABB's BNCI2014_001 + MotorImagery paradigm to fetch from the
    BNCI Horizon 2020 mirror (no registration required). Output matches
    the GDF converter: A0XT.npz (training session) and A0XE.npz (evaluation
    session) per subject, each with X: float32 [trials, 22, 1000], y: int64.

    The wide default fmin/fmax leaves data effectively unfiltered so the
    canonical 8-30 Hz bandpass in `BCI2aNPZ` runs at training time.

    Args:
        subjects: iterable of 1-based subject ids in 1..9.
        out_dir: output directory for .npz files.
        tmin, tmax: epoch window relative to cue onset (seconds).
        fmin, fmax: passband applied by MOABB during epoching.

    Returns:
        Dict mapping subject id to a summary dict.
    """
    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except ImportError as e:
        raise ImportError(
            "moabb is required for load_via_moabb. Install with `pip install moabb`."
        ) from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes=4, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, resample=None,
    )
    target_samples = int(round((tmax - tmin) * EXPECTED_SFREQ))

    summary: dict[int, dict] = {}
    for sid in subjects:
        X, y_str, meta = paradigm.get_data(dataset=dataset, subjects=[sid])

        if X.shape[1] != EXPECTED_EEG_CHANNELS:
            raise ValueError(
                f"subject {sid}: expected {EXPECTED_EEG_CHANNELS} channels, "
                f"got {X.shape[1]}"
            )

        y = np.array(
            [MOABB_LABEL_MAP[s] for s in np.asarray(y_str)], dtype=np.int64
        )

        if X.shape[-1] > target_samples:
            X = X[..., :target_samples]
        X = X.astype(np.float32)

        # Split into train/eval by session column. MOABB reports two sessions
        # per subject; the first chronologically is training, the second eval.
        sessions = sorted(meta["session"].unique())
        if len(sessions) != 2:
            raise RuntimeError(
                f"subject {sid}: expected 2 sessions, got {sessions}"
            )
        train_session, eval_session = sessions
        train_mask = (meta["session"] == train_session).to_numpy()
        eval_mask = (meta["session"] == eval_session).to_numpy()

        train_path = out_dir / f"A0{sid}T.npz"
        eval_path = out_dir / f"A0{sid}E.npz"
        np.savez_compressed(train_path, X=X[train_mask], y=y[train_mask])
        np.savez_compressed(eval_path, X=X[eval_mask], y=y[eval_mask])

        summary[sid] = {
            "train_trials": int(train_mask.sum()),
            "eval_trials": int(eval_mask.sum()),
            "n_channels": int(X.shape[1]),
            "n_samples": int(X.shape[-1]),
            "train_session": str(train_session),
            "eval_session": str(eval_session),
        }
        print(f"  subject A0{sid}: {summary[sid]}")

    return summary


def _parse_subjects(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        return list(range(lo, hi + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert BCI Competition IV 2a to .npz.",
    )
    parser.add_argument(
        "--source", choices=("gdf", "moabb"), default="gdf",
        help="data source: local GDFs (default) or MOABB auto-download",
    )
    parser.add_argument("--out-dir", required=True, help="output dir for .npz files")
    parser.add_argument(
        "--subjects", default="1-9", help='e.g. "1-9" or "1,3,5"',
    )
    parser.add_argument(
        "--gdf-dir", default=None,
        help="(gdf source) dir with A0{1..9}{T,E}.gdf",
    )
    parser.add_argument(
        "--labels-dir", default=None,
        help="(gdf source) dir with A0{1..9}E.mat (defaults to --gdf-dir)",
    )
    args = parser.parse_args()

    subjects = _parse_subjects(args.subjects)

    if args.source == "gdf":
        if not args.gdf_dir:
            parser.error("--gdf-dir is required when --source gdf")
        for s in subjects:
            print(f"subject A0{s}:")
            convert_subject(s, args.gdf_dir, args.out_dir, args.labels_dir)
    else:
        load_via_moabb(subjects, args.out_dir)


if __name__ == "__main__":
    _main()
