"""Tests for src/preprocessing/filters.py.

Run as a script:
    python -m tests.test_filters

Or via pytest:
    pytest tests/test_filters.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# Repo root on sys.path so `src.*` resolves when running standalone.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.preprocessing.filters import (  # noqa: E402
    bandpass_filter,
    instantaneous_phase,
    standardize_per_channel,
)


class TestInstantaneousPhase(unittest.TestCase):
    def test_preserves_shape_3d(self) -> None:
        x = np.random.randn(4, 22, 1000).astype(np.float32)
        phase = instantaneous_phase(x, sampling_rate=250)
        self.assertEqual(phase.shape, x.shape)
        self.assertEqual(phase.dtype, np.float32)

    def test_preserves_shape_2d(self) -> None:
        x = np.random.randn(22, 1000).astype(np.float32)
        phase = instantaneous_phase(x, sampling_rate=250)
        self.assertEqual(phase.shape, x.shape)

    def test_phase_range(self) -> None:
        x = np.random.randn(2, 4, 500).astype(np.float32)
        phase = instantaneous_phase(x, sampling_rate=250)
        self.assertTrue(np.all(phase >= -np.pi - 1e-5))
        self.assertTrue(np.all(phase <= np.pi + 1e-5))

    def test_pure_sinusoid_advances_linearly(self) -> None:
        """For a clean 10 Hz sinusoid the unwrapped phase should advance at
        2*pi*f/fs per sample. Bandpass + Hilbert has small edge artifacts,
        so we evaluate the slope on the middle of the window only."""
        fs, freq, n = 250, 10.0, 1000
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * freq * t).astype(np.float32)

        phase = instantaneous_phase(x, sampling_rate=fs, low=8.0, high=30.0)
        unwrapped = np.unwrap(phase)

        mid = slice(n // 4, 3 * n // 4)  # avoid Hilbert edge artifacts
        slope = np.polyfit(np.arange(n)[mid], unwrapped[mid], 1)[0]
        expected = 2 * np.pi * freq / fs
        self.assertAlmostEqual(slope, expected, places=3)

    def test_skipping_filter(self) -> None:
        x = np.random.randn(2, 100).astype(np.float32)
        phase_filtered = instantaneous_phase(x, sampling_rate=250)
        phase_raw = instantaneous_phase(x, sampling_rate=250, low=None, high=None)
        # Both should be valid phase arrays of matching shape;
        # values differ because filtering changes the signal.
        self.assertEqual(phase_filtered.shape, phase_raw.shape)
        self.assertTrue(np.all(np.abs(phase_raw) <= np.pi + 1e-5))

    def test_half_specified_band_raises(self) -> None:
        x = np.random.randn(2, 100).astype(np.float32)
        with self.assertRaises(ValueError):
            instantaneous_phase(x, sampling_rate=250, low=8.0, high=None)


class TestBandpassFilter(unittest.TestCase):
    def test_attenuates_out_of_band(self) -> None:
        """A 50 Hz signal should be sharply attenuated by an 8-30 Hz filter."""
        fs, n = 250, 2000
        t = np.arange(n) / fs
        in_band = np.sin(2 * np.pi * 15 * t)
        out_band = np.sin(2 * np.pi * 50 * t)
        x = (in_band + out_band).astype(np.float32)

        filtered = bandpass_filter(x, fs, low=8.0, high=30.0)
        # Energy in middle window — passband signal should dominate.
        mid = slice(n // 4, 3 * n // 4)
        passband_energy = np.mean(filtered[mid] ** 2)
        original_energy = np.mean(in_band[mid] ** 2)
        # Filtered signal preserves >80% of in-band energy.
        self.assertGreater(passband_energy, 0.8 * original_energy)
        # And rejects most out-of-band energy (well below 1.0).
        self.assertLess(passband_energy, 1.2 * original_energy)


class TestStandardize(unittest.TestCase):
    def test_zero_mean_unit_var(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((2, 4, 500)).astype(np.float32) * 10 + 5
        z = standardize_per_channel(x)
        self.assertTrue(np.allclose(z.mean(axis=-1), 0, atol=1e-5))
        self.assertTrue(np.allclose(z.std(axis=-1), 1, atol=1e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
