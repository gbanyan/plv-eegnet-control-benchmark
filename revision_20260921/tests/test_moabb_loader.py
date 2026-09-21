"""Smoke test for the MOABB-backed loader.

Mocks `MotorImagery.get_data` so the test never hits the network. Verifies
that `load_via_moabb` writes A0XT.npz and A0XE.npz with the right shape,
dtype, label range, and per-session split.

Run as a script:
    python -m tests.test_moabb_loader

Or via pytest:
    pytest tests/test_moabb_loader.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _have(*mods: str) -> bool:
    for m in mods:
        try:
            __import__(m)
        except ImportError:
            return False
    return True


def _fake_paradigm_get_data(
    n_train: int = 12, n_eval: int = 8, n_channels: int = 22, n_samples: int = 1001,
):
    """Build a fake `(X, y, meta)` tuple matching MOABB's MotorImagery output."""
    import pandas as pd

    rng = np.random.default_rng(0)
    n_trials = n_train + n_eval
    X = rng.standard_normal((n_trials, n_channels, n_samples)).astype(np.float64)

    classes = ["left_hand", "right_hand", "feet", "tongue"]
    y = np.array([classes[i % 4] for i in range(n_trials)])

    sessions = ["0train"] * n_train + ["1test"] * n_eval
    meta = pd.DataFrame({
        "subject": [1] * n_trials,
        "session": sessions,
        "run": ["run_0"] * n_trials,
    })
    return X, y, meta


@unittest.skipUnless(_have("moabb", "pandas"), "moabb and pandas are required")
class TestLoadViaMoabb(unittest.TestCase):
    def setUp(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        # Patch BOTH the dataset constructor (no network) and the paradigm's
        # get_data (no real epoching). We patch via the module namespace where
        # `load_via_moabb` looks them up: `moabb.datasets` and `moabb.paradigms`.
        import moabb.datasets as md
        import moabb.paradigms as mp

        self._orig_dataset_cls = md.BNCI2014_001
        self._orig_get_data = mp.MotorImagery.get_data

        class _StubDataset:
            subject_list = list(range(1, 10))
            code = "BNCI2014-001"

            def __init__(self):
                pass

        md.BNCI2014_001 = _StubDataset
        mp.MotorImagery.get_data = lambda self, dataset, subjects=None, **kw: (
            _fake_paradigm_get_data()
        )

        self._md = md
        self._mp = mp
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self._md.BNCI2014_001 = self._orig_dataset_cls
        self._mp.MotorImagery.get_data = self._orig_get_data
        self.tmp.cleanup()

    def test_writes_train_and_eval_npz_per_subject(self) -> None:
        from src.datasets import bci2a

        summary = bci2a.load_via_moabb(subjects=[1, 2], out_dir=self.tmp_path)

        for sid in (1, 2):
            train = self.tmp_path / f"A0{sid}T.npz"
            eval_ = self.tmp_path / f"A0{sid}E.npz"
            self.assertTrue(train.exists(), f"missing {train}")
            self.assertTrue(eval_.exists(), f"missing {eval_}")

            with np.load(train) as d:
                Xt, yt = d["X"], d["y"]
            with np.load(eval_) as d:
                Xe, ye = d["X"], d["y"]

            # Shapes: 12 train + 8 eval, 22 channels, trimmed from 1001 -> 1000.
            self.assertEqual(Xt.shape, (12, 22, 1000))
            self.assertEqual(Xe.shape, (8, 22, 1000))
            self.assertEqual(Xt.dtype, np.float32)
            self.assertEqual(Xe.dtype, np.float32)

            # Labels live in {0, 1, 2, 3} and use int64.
            self.assertEqual(yt.dtype, np.int64)
            self.assertEqual(ye.dtype, np.int64)
            self.assertTrue(set(yt.tolist()).issubset({0, 1, 2, 3}))
            self.assertTrue(set(ye.tolist()).issubset({0, 1, 2, 3}))

            self.assertEqual(summary[sid]["train_trials"], 12)
            self.assertEqual(summary[sid]["eval_trials"], 8)
            self.assertEqual(summary[sid]["n_channels"], 22)
            self.assertEqual(summary[sid]["n_samples"], 1000)
            # Session ordering: "0train" sorts before "1test".
            self.assertEqual(summary[sid]["train_session"], "0train")
            self.assertEqual(summary[sid]["eval_session"], "1test")

    def test_rejects_wrong_channel_count(self) -> None:
        # Override the stub to return a 25-channel tensor (raw + EOG).
        self._mp.MotorImagery.get_data = lambda self, dataset, subjects=None, **kw: (
            _fake_paradigm_get_data(n_channels=25)
        )
        from src.datasets import bci2a

        with self.assertRaises(ValueError):
            bci2a.load_via_moabb(subjects=[1], out_dir=self.tmp_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
