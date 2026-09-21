"""Phase-aware EEGNet.

EEGNet with the Hilbert *quadrature* (sin of instantaneous phase)
concatenated as additional input channels. The depthwise spatial conv
then sees both the raw signal AND a phase-shifted (90 deg) version of
the same band.

Important: cos(theta) is NOT added separately because the bandpassed raw
signal is essentially A(t)*cos(theta(t)) — adding cos(theta) gives ~90%
collinear inputs, which destabilizes optimization. sin(theta) is
orthogonal in the analytic-signal sense and carries the new info.

The depthwise spatial conv learns:

    y[f, t] = sum_c W_raw[f, c] * x[c, t] + W_sin[f, c] * sin(theta[c, t])

so it can express phase-rotated weighted spatial filters. Linear conv
alone cannot represent full phase *coupling* (cos(theta_i - theta_j))
since that's a product across channels — but exposing phase-rotated
features gives the model a richer input space than raw EEG alone.

Ablation control: `phase_mode='shuffle'` time-permutes the phase tensor
per (batch, channel), preserving the marginal distribution while breaking
temporal alignment with the raw signal. If real phase beats shuffled
phase, the gain comes from phase *structure*, not extra channel capacity.

Input shape: [batch, channels, time].
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from src.models.eegnet import EEGNet
from src.preprocessing.filters import analytic_signal_torch

PhaseMode = Literal["phase", "shuffle"]


class PhaseEEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        n_samples: int = 1000,
        sampling_rate: int = 250,
        phase_band: tuple[float, float] | None = (8.0, 30.0),
        phase_mode: PhaseMode = "phase",
        # Standard EEGNet knobs forwarded to the inner model.
        temporal_kernel: int | None = 125,
        F1: int = 8,
        D: int = 2,
        F2: int | None = 16,
        dropout: float = 0.5,
    ):
        super().__init__()
        if phase_mode not in ("phase", "shuffle"):
            raise ValueError(f"phase_mode must be 'phase' or 'shuffle', got {phase_mode!r}")

        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.phase_band = phase_band
        self.phase_mode = phase_mode

        # 2x channels: raw, sin(theta). cos(theta) is omitted because it's
        # ~90% collinear with the raw signal in this band.
        self.eegnet = EEGNet(
            n_channels=n_channels * 2,
            n_classes=n_classes,
            n_samples=n_samples,
            sampling_rate=sampling_rate,
            temporal_kernel=temporal_kernel,
            F1=F1, D=D, F2=F2, dropout=dropout,
        )

    def _quadrature(self, x: torch.Tensor) -> torch.Tensor:
        """Return sin(theta), shape [B, C, T]. The orthogonal (quadrature)
        component to the raw bandpassed signal."""
        low, high = (None, None) if self.phase_band is None else self.phase_band
        with torch.no_grad():
            analytic = analytic_signal_torch(
                x, sampling_rate=self.sampling_rate, low=low, high=high,
            )
            phase = torch.angle(analytic)  # [B, C, T]

            if self.phase_mode == "shuffle":
                # Independent random time-permutation per (batch, channel).
                # Preserves marginal distribution; destroys temporal alignment.
                B, C, T = phase.shape
                perm = torch.rand(B, C, T, device=phase.device).argsort(dim=-1)
                phase = torch.gather(phase, -1, perm)

            return torch.sin(phase)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected 3D input [B, C, T], got shape {tuple(x.shape)}")
        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} channels, got {x.shape[1]}"
            )

        quad = self._quadrature(x)
        x_aug = torch.cat([x, quad], dim=1)  # [B, 2C, T]
        return self.eegnet(x_aug)
