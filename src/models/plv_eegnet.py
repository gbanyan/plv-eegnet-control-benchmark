"""EEGNet + Phase-Locking Value (PLV) auxiliary features.

PLV between channels i and j over a window is:

    PLV_ij = | (1/T) * sum_t exp( i * (theta_i(t) - theta_j(t)) ) |    in [0, 1]

PLV captures *windowed coupling* — whether two signals consistently maintain
a phase relationship over the trial — which is fundamentally different from
the instantaneous cos(theta_i - theta_j) used in earlier phase-attention
attempts. This is the BCI-literature standard for "phase coupling matters."

Architecture:
    [B, 22, 1000]
        -> EEGNet trunk ----------------------------> flat features [B, F2*T_pool]
        -> phase -> pairwise PLV (upper triangle) --> BN -> [B, n_pairs]
                                                concat
                                                  v
                                          Linear(flat + n_pairs, n_classes)

Modes:
    'real'           - compute PLV from real instantaneous phase
    'zero'           - feed zeros in place of PLV features (controls for capacity)
    'shuffle_pairs'  - real PLV values, but per-trial random permutation of the
                       n_pairs dimension. Same value distribution as 'real',
                       same capacity, but channel-pair identity is destroyed.
                       Tests whether the model uses *which* pair couples to
                       which (structure), not just the distribution of
                       coupling magnitudes. NOTE: the permutation is redrawn on
                       every forward pass, so across epochs a given trial sees a
                       fresh permutation (an implicit stochastic regularizer).
    'shuffle_pairs_fixed' - same as 'shuffle_pairs' but each trial's permutation
                       is derived deterministically from a content hash of its
                       own input, so it is STABLE across epochs (and distinct
                       between trials). Isolates the same-marginal pair-position
                       effect from the per-epoch reshuffling stochastic
                       regularizer, without plumbing trial indices through the
                       training loop.

Input shape: [batch, channels, time].
"""

from __future__ import annotations

import hashlib
from typing import Literal

import torch
import torch.nn as nn

from src.models.eegnet import EEGNet
from src.preprocessing.filters import instantaneous_phase_torch

PLVMode = Literal["real", "zero", "shuffle_pairs", "shuffle_pairs_fixed", "pli", "wpli"]


class PLVEEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        n_samples: int = 1000,
        sampling_rate: int = 250,
        phase_band: tuple[float, float] | None = (8.0, 30.0),
        plv_mode: PLVMode = "real",
        bottleneck_dim: int | None = None,
        fixed_perm_salt: int = 0,
        # Standard EEGNet knobs forwarded to the trunk.
        temporal_kernel: int | None = 125,
        F1: int = 8,
        D: int = 2,
        F2: int | None = 16,
        dropout: float = 0.5,
    ):
        super().__init__()
        if plv_mode not in ("real", "zero", "shuffle_pairs", "shuffle_pairs_fixed", "pli", "wpli"):
            raise ValueError(
                "plv_mode must be 'real', 'zero', 'shuffle_pairs', 'shuffle_pairs_fixed', "
                f"'pli', or 'wpli', got {plv_mode!r}"
            )

        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.phase_band = phase_band
        self.plv_mode = plv_mode
        self.fixed_perm_salt = int(fixed_perm_salt)
        self._perm_key = str(self.fixed_perm_salt).encode()

        # Number of unique channel pairs (upper triangle, no diagonal).
        self.n_pairs = n_channels * (n_channels - 1) // 2

        # Use EEGNet as the trunk; we'll bypass its final classifier and add
        # our own that takes [eeg_flat + n_pairs] features.
        self.eegnet = EEGNet(
            n_channels=n_channels, n_classes=n_classes, n_samples=n_samples,
            sampling_rate=sampling_rate, temporal_kernel=temporal_kernel,
            F1=F1, D=D, F2=F2, dropout=dropout,
        )
        # Trick: introspect the trunk's flattened feature size from its own
        # already-built classifier.
        eeg_flat = self.eegnet.classifier.in_features

        # Replace the inner classifier with identity-equivalent so the trunk
        # only contributes flat features.
        self.eegnet.classifier = nn.Identity()  # type: ignore[assignment]

        self.plv_norm = nn.BatchNorm1d(self.n_pairs)
        # Optional bottleneck projection that compresses the n_pairs PLV
        # vector before concatenation. Tests whether the (real - shuffle)
        # gap is dimensionality-driven (P >> N overfitting interaction with
        # FC layer) vs content-driven (pair-indexed coupling structure).
        self.bottleneck_dim = bottleneck_dim
        if bottleneck_dim is None:
            self.plv_proj: nn.Module = nn.Identity()
            plv_out_dim = self.n_pairs
        else:
            self.plv_proj = nn.Linear(self.n_pairs, bottleneck_dim)
            plv_out_dim = bottleneck_dim
        self.classifier = nn.Linear(eeg_flat + plv_out_dim, n_classes)

        # Cache upper-triangle indices once.
        iu = torch.triu_indices(n_channels, n_channels, offset=1)
        self.register_buffer("_triu_i", iu[0], persistent=False)
        self.register_buffer("_triu_j", iu[1], persistent=False)

    def _compute_plv(self, x: torch.Tensor) -> torch.Tensor:
        """Pairwise coupling features over the trial window. Returns [B, n_pairs].

        Behavior depends on `plv_mode`:
          - 'zero': all-zero vector (architecture-overhead control)
          - 'real' / 'shuffle_pairs': PLV via einsum identity (avoids
            [B, C, C, T] intermediate)
          - 'pli': phase-lag index, computed pair-by-pair in channel chunks
            because sign() prevents the einsum identity from working
          - 'wpli': weighted phase-lag index (Vinck et al. 2011), debiased
            volume-conduction-resistant variant of PLI; also chunked
        """
        if self.plv_mode == "zero":
            return torch.zeros(x.shape[0], self.n_pairs, device=x.device, dtype=x.dtype)

        low, high = (None, None) if self.phase_band is None else self.phase_band
        with torch.no_grad():
            phase = instantaneous_phase_torch(
                x, sampling_rate=self.sampling_rate, low=low, high=high,
            )  # [B, C, T]
            B, C, T = phase.shape
            c = torch.cos(phase)  # [B, C, T]
            s = torch.sin(phase)  # [B, C, T]

            if self.plv_mode in ("pli", "wpli"):
                # sin(phi_i - phi_j) = s_i c_j - c_i s_j; build chunked.
                #   PLI:  | E_t [ sign( sin diff ) ] |
                #   wPLI: | E_t [ sin diff ] | / E_t [ |sin diff| ]
                #         (Vinck et al. 2011; bounded in [0, 1]; equals 0
                #          when phase-difference distribution is symmetric
                #          around 0, which is the volume-conduction signature.)
                # Both build the full [B, C, C] matrix in row chunks to avoid
                # the [B, C, C, T] tensor at high channel counts.
                out_matrix = torch.zeros(B, C, C, device=x.device, dtype=x.dtype)
                chunk_size = 16
                for start in range(0, C, chunk_size):
                    end = min(start + chunk_size, C)
                    s_i = s[:, start:end].unsqueeze(2)  # [B, k, 1, T]
                    c_i = c[:, start:end].unsqueeze(2)
                    s_j = s.unsqueeze(1)  # [B, 1, C, T]
                    c_j = c.unsqueeze(1)
                    sin_diff = s_i * c_j - c_i * s_j   # [B, k, C, T]
                    if self.plv_mode == "pli":
                        out_matrix[:, start:end] = sin_diff.sign().mean(dim=-1).abs()
                    else:  # wpli
                        num = sin_diff.mean(dim=-1).abs()              # [B, k, C]
                        den = sin_diff.abs().mean(dim=-1).clamp_min(1e-8)
                        out_matrix[:, start:end] = num / den
                out = out_matrix[:, self._triu_i, self._triu_j]
                return out

            # Real PLV via einsum identity:
            #   PLV_ij = | mean_t (cos d + i sin d) |
            #          = sqrt( (mean_t cos d)^2 + (mean_t sin d)^2 )
            # where d = phi_i - phi_j. Avoid the [B, C, C, T] intermediate by
            # noting cos(a - b) = cos a cos b + sin a sin b, sin(a - b) = sin a cos b - cos a sin b,
            # so mean_t cos(phi_i - phi_j) = (1/T) [c @ c^T + s @ s^T]_{ij}
            # and  mean_t sin(phi_i - phi_j) = (1/T) [s @ c^T - c @ s^T]_{ij}.
            cct = torch.einsum("bct,bdt->bcd", c, c) / T  # [B, C, C]
            sst = torch.einsum("bct,bdt->bcd", s, s) / T
            sct = torch.einsum("bct,bdt->bcd", s, c) / T
            cst = torch.einsum("bct,bdt->bcd", c, s) / T
            mean_cos = cct + sst
            mean_sin = sct - cst
            plv_matrix = torch.sqrt(mean_cos * mean_cos + mean_sin * mean_sin)

            # Upper triangle (no diagonal) -> [B, n_pairs]
            plv = plv_matrix[:, self._triu_i, self._triu_j]

            if self.plv_mode == "shuffle_pairs":
                # Independent per-trial random permutation of the n_pairs
                # dimension. Same value distribution, channel-pair identity
                # destroyed. Applied at both train and eval time so the
                # classifier never gets a stable per-pair signal.
                B, P = plv.shape
                perm = torch.rand(B, P, device=plv.device).argsort(dim=-1)
                plv = torch.gather(plv, -1, perm)
            elif self.plv_mode == "shuffle_pairs_fixed":
                # Same marginal as 'shuffle_pairs', but each trial's permutation
                # is keyed to a content hash of its own (quantized) phase
                # features, so it is STABLE across epochs and distinct between
                # trials. This removes the per-epoch reshuffling stochastic
                # regularizer while still destroying channel-pair identity.
                # Quantization makes the fingerprint robust to last-bit FP noise.
                B, P = plv.shape
                fp = (c.reshape(B, -1)[:, :512] * 1e4).round().to(torch.int32).cpu().numpy()
                perms = torch.empty(B, P, dtype=torch.long)
                for b in range(B):
                    seed = int.from_bytes(
                        hashlib.blake2b(fp[b].tobytes(), key=self._perm_key, digest_size=8).digest(), "little"
                    ) % (2 ** 31)
                    g = torch.Generator().manual_seed(seed)
                    perms[b] = torch.randperm(P, generator=g)
                plv = torch.gather(plv, -1, perms.to(plv.device))

            return plv

    def _eegnet_features(self, x: torch.Tensor) -> torch.Tensor:
        # Replicate the trunk's forward up through flatten, but skip the
        # (now-Identity) classifier.
        if x.dim() != 3:
            raise ValueError(f"expected 3D input [B, C, T], got shape {tuple(x.shape)}")
        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} channels, got {x.shape[1]}"
            )
        h = x.unsqueeze(1)
        h = self.eegnet._forward_features(h)
        return h.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._eegnet_features(x)              # [B, eeg_flat]
        plv = self._compute_plv(x)                   # [B, n_pairs]
        plv = self.plv_norm(plv)
        plv = self.plv_proj(plv)                     # Identity or Linear(P, bn_dim)
        return self.classifier(torch.cat([feat, plv], dim=1))
