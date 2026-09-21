"""Smoke test for the BCI 2a GDF -> NPZ converter.

We never read a real GDF here. Instead we build an in-memory `mne.io.RawArray`
with annotations that mimic the BCI 2a protocol (trial-start 768, cues
769-772, artifact 1023) and monkey-patch `mne.io.read_raw_gdf` to return it.

Run as a script:
    python -m tests.test_bci2a_converter

Or via pytest:
    pytest tests/test_bci2a_converter.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _have_mne() -> bool:
    try:
        import mne  # noqa: F401
        return True
    except ImportError:
        return False


def _build_synthetic_raw(
    sfreq: int = 250,
    cue_codes: tuple[str, ...] = ("769", "770", "771", "772"),
    artifact_on_trials: tuple[int, ...] = (0,),
    trial_spacing_s: float = 8.0,
    eog_named: bool = True,
):
    """Build a synthetic BCI-2a-shaped RawArray with 22 EEG + 3 EOG channels.

    One trial per cue code. The first trial is artifact-marked unless
    `artifact_on_trials` is overridden. If `eog_named` is False, the EOG
    channels keep generic names (EOG type only) — this is what some MNE
    versions produce when reading real BCI 2a GDFs, so it exercises the
    type-based EOG-drop path.
    """
    import mne

    n_eeg, n_eog = 22, 3
    eog_names = (
        ["EOG-left", "EOG-central", "EOG-right"] if eog_named
        else ["x-aux-1", "x-aux-2", "x-aux-3"]  # name doesn't say EOG
    )
    ch_names = [f"EEG{i:02d}" for i in range(n_eeg)] + eog_names
    ch_types = ["eeg"] * n_eeg + ["eog"] * n_eog
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    n_trials = len(cue_codes)
    total_s = trial_spacing_s * n_trials + 6.0
    n_samples = int(total_s * sfreq)

    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_eeg + n_eog, n_samples)).astype(np.float64) * 1e-6

    raw = mne.io.RawArray(data, info, verbose="ERROR")

    onsets, durations, descriptions = [], [], []
    for i, code in enumerate(cue_codes):
        trial_start_s = 1.0 + i * trial_spacing_s
        cue_s = trial_start_s + 2.0  # cue is 2 s after trial start in BCI 2a
        onsets.append(trial_start_s); durations.append(0.0); descriptions.append("768")
        if i in artifact_on_trials:
            onsets.append(trial_start_s); durations.append(0.0); descriptions.append("1023")
        onsets.append(cue_s); durations.append(0.0); descriptions.append(code)

    raw.set_annotations(mne.Annotations(onsets, durations, descriptions))
    return raw


@unittest.skipUnless(_have_mne(), "mne is not installed")
class TestBCI2aConverter(unittest.TestCase):
    def setUp(self) -> None:
        # Imported here so the skip decorator can run without mne installed.
        import mne
        from src.datasets import bci2a

        self.mne = mne
        self.bci2a = bci2a
        self._orig_read = mne.io.read_raw_gdf

        # Repo root on sys.path so `src.*` resolves when running standalone.
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.mne.io.read_raw_gdf = self._orig_read
        self.tmp.cleanup()

    # --- training file ----------------------------------------------------

    def test_train_conversion_drops_artifact_trial(self) -> None:
        raw = _build_synthetic_raw(artifact_on_trials=(0,))
        self.mne.io.read_raw_gdf = lambda path, **kw: raw

        fake_gdf = self.tmp_path / "A01T.gdf"
        fake_gdf.touch()
        out = self.tmp_path / "A01T.npz"

        info = self.bci2a.convert_gdf_to_npz(fake_gdf, out)

        self.assertTrue(out.exists())
        with np.load(out) as data:
            X, y = data["X"], data["y"]

        # 4 cues, 1 trial dropped for artifact -> 3 surviving trials.
        self.assertEqual(X.shape, (3, 22, 1000))
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.shape, (3,))
        # Trial 0 (label 0 = left hand) is the dropped one.
        self.assertEqual(set(y.tolist()), {1, 2, 3})
        self.assertEqual(info["n_dropped"], 1)
        self.assertEqual(info["n_channels"], 22)
        self.assertEqual(info["sfreq"], 250)

    def test_train_conversion_drops_eog_by_type_not_name(self) -> None:
        """Real BCI 2a GDFs sometimes give EOG channels generic names; the
        converter must drop them by channel *type*, leaving 22 EEG channels."""
        raw = _build_synthetic_raw(artifact_on_trials=(), eog_named=False)
        self.mne.io.read_raw_gdf = lambda path, **kw: raw

        fake_gdf = self.tmp_path / "A01T.gdf"
        fake_gdf.touch()
        out = self.tmp_path / "A01T.npz"

        info = self.bci2a.convert_gdf_to_npz(fake_gdf, out)
        self.assertEqual(info["n_channels"], 22)
        with np.load(out) as data:
            self.assertEqual(data["X"].shape, (4, 22, 1000))

    def test_wrong_sample_rate_raises(self) -> None:
        raw = _build_synthetic_raw(sfreq=500, artifact_on_trials=())
        self.mne.io.read_raw_gdf = lambda path, **kw: raw

        fake_gdf = self.tmp_path / "A01T.gdf"
        fake_gdf.touch()
        out = self.tmp_path / "A01T.npz"

        with self.assertRaises(ValueError):
            self.bci2a.convert_gdf_to_npz(fake_gdf, out)

    def test_train_conversion_keeps_all_when_no_artifacts(self) -> None:
        raw = _build_synthetic_raw(artifact_on_trials=())
        self.mne.io.read_raw_gdf = lambda path, **kw: raw

        fake_gdf = self.tmp_path / "A01T.gdf"
        fake_gdf.touch()
        out = self.tmp_path / "A01T.npz"

        info = self.bci2a.convert_gdf_to_npz(fake_gdf, out)
        with np.load(out) as data:
            X, y = data["X"], data["y"]

        self.assertEqual(X.shape, (4, 22, 1000))
        self.assertEqual(sorted(y.tolist()), [0, 1, 2, 3])
        self.assertEqual(info["n_dropped"], 0)

    # --- evaluation file --------------------------------------------------

    def test_eval_conversion_uses_classlabel_mat(self) -> None:
        from scipy.io import savemat

        # All cues anonymized to 783; true labels come from the .mat file.
        raw = _build_synthetic_raw(
            cue_codes=("783", "783", "783", "783"),
            artifact_on_trials=(),
        )
        self.mne.io.read_raw_gdf = lambda path, **kw: raw

        fake_gdf = self.tmp_path / "A01E.gdf"
        fake_gdf.touch()
        labels_mat = self.tmp_path / "A01E.mat"
        # MAT classlabel uses 1-based labels; converter shifts to 0-based.
        savemat(labels_mat, {"classlabel": np.array([[1], [2], [3], [4]])})
        out = self.tmp_path / "A01E.npz"

        self.bci2a.convert_gdf_to_npz(fake_gdf, out, labels_mat_path=labels_mat)

        with np.load(out) as data:
            X, y = data["X"], data["y"]
        self.assertEqual(X.shape, (4, 22, 1000))
        self.assertEqual(y.tolist(), [0, 1, 2, 3])

    def test_eval_without_mat_raises(self) -> None:
        raw = _build_synthetic_raw(
            cue_codes=("783",), artifact_on_trials=()
        )
        self.mne.io.read_raw_gdf = lambda path, **kw: raw

        fake_gdf = self.tmp_path / "A01E.gdf"
        fake_gdf.touch()
        out = self.tmp_path / "A01E.npz"

        with self.assertRaises(ValueError):
            self.bci2a.convert_gdf_to_npz(fake_gdf, out)


if __name__ == "__main__":
    if not _have_mne():
        print("[skip] mne is not installed; install with `pip install mne scipy`")
        sys.exit(0)
    unittest.main(verbosity=2)
