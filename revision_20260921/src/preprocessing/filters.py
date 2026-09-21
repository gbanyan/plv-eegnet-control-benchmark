"""Signal preprocessing utilities for EEG."""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import butter, filtfilt, hilbert, iirnotch


def bandpass_filter(
    x: np.ndarray,
    sampling_rate: int,
    low: float = 8.0,
    high: float = 30.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter applied along the last axis.

    Args:
        x: array of shape [..., time].
        sampling_rate: sampling rate in Hz.
        low, high: passband edges in Hz.
        order: filter order.
    """
    nyq = 0.5 * sampling_rate
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x, axis=-1).astype(np.float32, copy=False)


def notch_filter(
    x: np.ndarray,
    sampling_rate: int,
    freq: float,
    q: float = 30.0,
) -> np.ndarray:
    """Zero-phase IIR notch at `freq` Hz (and at the harmonic if it
    fits below Nyquist), applied along the last axis."""
    out = x
    nyq = 0.5 * sampling_rate
    f = freq
    while f < nyq - 1.0:
        b, a = iirnotch(f / nyq, q)
        out = filtfilt(b, a, out, axis=-1)
        f += freq
    return out.astype(np.float32, copy=False)


def instantaneous_phase(
    x: np.ndarray,
    sampling_rate: int,
    low: float | None = 8.0,
    high: float | None = 30.0,
    order: int = 4,
) -> np.ndarray:
    """Per-channel instantaneous phase via the Hilbert transform.

    Pipeline: optional bandpass -> analytic signal -> angle. Phase is
    computed along the last axis, so input shape `[..., time]` is preserved.
    Output values are in (-pi, pi].

    Args:
        x: array of shape [..., time].
        sampling_rate: sampling rate in Hz.
        low, high: bandpass edges in Hz. Pass `low=None` and `high=None`
            (or both) to skip filtering. Default is the canonical 8-30 Hz
            sensorimotor band used elsewhere in this project.
        order: Butterworth order for the bandpass.

    Returns:
        float32 array of phases, same shape as `x`.
    """
    if (low is None) ^ (high is None):
        raise ValueError("low and high must be both set or both None")
    signal = x
    if low is not None and high is not None:
        signal = bandpass_filter(signal, sampling_rate, low=low, high=high, order=order)
    analytic = hilbert(signal, axis=-1)
    return np.angle(analytic).astype(np.float32, copy=False)


def analytic_signal_torch(
    x: torch.Tensor,
    sampling_rate: int,
    low: float | None = 8.0,
    high: float | None = 30.0,
) -> torch.Tensor:
    """FFT-based bandpass + Hilbert in a single shot, on the input device.

    Returns the (complex) analytic signal restricted to [low, high]. Take
    `torch.angle(...)` for instantaneous phase. Operates along the last
    axis. Shape preserved.

    The Hilbert transform comes for free here: zeroing the negative
    frequency components and doubling the positive band gives the analytic
    signal. So we combine bandpass + Hilbert into one FFT mask.
    """
    if (low is None) ^ (high is None):
        raise ValueError("low and high must be both set or both None")

    n = x.shape[-1]
    X = torch.fft.fft(x, dim=-1)
    freqs = torch.fft.fftfreq(n, d=1.0 / sampling_rate, device=x.device)

    # Default Hilbert mask: zero negative freqs, double positive freqs.
    mask = torch.zeros(n, dtype=x.dtype, device=x.device)
    mask[freqs > 0] = 2.0
    mask[freqs == 0] = 1.0
    if n % 2 == 0:
        mask[n // 2] = 1.0  # Nyquist component

    # Optional bandpass: also zero anything outside the passband.
    if low is not None and high is not None:
        in_band = (freqs.abs() >= low) & (freqs.abs() <= high)
        mask = mask * in_band.to(mask.dtype)

    return torch.fft.ifft(X * mask, dim=-1)


def instantaneous_phase_torch(
    x: torch.Tensor,
    sampling_rate: int,
    low: float | None = 8.0,
    high: float | None = 30.0,
) -> torch.Tensor:
    """Instantaneous phase, in (-pi, pi], computed on the input device."""
    return torch.angle(analytic_signal_torch(x, sampling_rate, low=low, high=high))


def standardize_per_channel(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Zero-mean, unit-variance per channel.

    Args:
        x: [trials, channels, time] or [channels, time].
    """
    if x.ndim == 2:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
    elif x.ndim == 3:
        # Per (trial, channel) normalization across time.
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
    else:
        raise ValueError(f"expected 2D or 3D array, got {x.ndim}D")
    return ((x - mean) / (std + eps)).astype(np.float32, copy=False)
