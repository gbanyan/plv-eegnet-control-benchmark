"""Phase-biased Transformer (Phase 2c).

Identical to TinyTransformer (Phase 2a) except attention scores carry a
phase-coherence bias:

    score_ij = (Q_i K_j^T) / sqrt(d) + lambda_h * cos(theta_i - theta_j)

where theta is the instantaneous phase per token (circular mean of
8-30 Hz phase across channels and patch samples), and lambda_h is a
per-head learnable scalar initialized to 0. With lambda == 0 frozen, this
model is bit-exact equivalent to TinyTransformer — that is the canonical
ablation for attributing any gain to the phase term.

Input shape: [batch, channels, time].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.preprocessing.filters import instantaneous_phase_torch


class PhaseBiasedAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 lam_init: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Per-head phase-bias coefficient. Init at 0 so the model starts
        # behaving exactly like a plain Transformer.
        self.lam = nn.Parameter(torch.full((n_heads,), float(lam_init)))

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        # x:     [B, N, D]
        # phase: [B, N]   (single scalar phase per token)
        B, N, D = x.shape
        H, d = self.n_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)              # each [B, H, N, d]

        scale = 1.0 / (d ** 0.5)
        score = (q @ k.transpose(-1, -2)) * scale  # [B, H, N, N]

        # cos(theta_i - theta_j) is symmetric and shared across heads.
        # Use the trig identity to keep two unsqueezes simple.
        delta = phase.unsqueeze(2) - phase.unsqueeze(1)   # [B, N, N]
        cos_delta = torch.cos(delta).unsqueeze(1)         # [B, 1, N, N]
        score = score + self.lam.view(1, H, 1, 1) * cos_delta

        attn = F.softmax(score, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v                       # [B, H, N, d]
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj_dropout(self.out(out))


class PhaseEncoderLayer(nn.Module):
    """Pre-norm encoder block: x + Attn(LN(x)) ; x + FF(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int,
                 dropout: float = 0.0, lam_init: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PhaseBiasedAttention(d_model, n_heads, dropout, lam_init=lam_init)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), phase)
        x = x + self.ff(self.norm2(x))
        return x


class PhaseTransformer(nn.Module):
    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        n_samples: int = 1000,
        patch_size: int = 25,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.5,
        sampling_rate: int = 250,
        phase_band: tuple[float, float] | None = (8.0, 30.0),
        lam_init: float = 0.0,
        freeze_lam: bool = False,
    ):
        super().__init__()
        if n_samples % patch_size != 0:
            raise ValueError(
                f"n_samples ({n_samples}) must be divisible by patch_size ({patch_size})"
            )

        self.n_channels = n_channels
        self.patch_size = patch_size
        self.sampling_rate = sampling_rate
        self.phase_band = phase_band
        n_patches = n_samples // patch_size

        self.patch_embed = nn.Conv1d(
            n_channels, d_model, kernel_size=patch_size, stride=patch_size,
        )
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            PhaseEncoderLayer(d_model, n_heads, ff_dim, dropout, lam_init=lam_init)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

        if freeze_lam:
            for layer in self.layers:
                layer.attn.lam.requires_grad_(False)

    def _phase_per_token(self, x: torch.Tensor) -> torch.Tensor:
        """Compute one phase scalar per (batch, patch) by circular-mean of
        instantaneous phase across (channels, patch_samples)."""
        low, high = (None, None) if self.phase_band is None else self.phase_band
        # Phase computation is deterministic w.r.t. x and not part of the
        # backward pass — detaching prevents wasted autograd bookkeeping.
        with torch.no_grad():
            phase_per_sample = instantaneous_phase_torch(
                x, sampling_rate=self.sampling_rate, low=low, high=high,
            )                                          # [B, C, T]
            B, C, T = phase_per_sample.shape
            n_patches = T // self.patch_size
            ps = phase_per_sample.reshape(B, C, n_patches, self.patch_size)
            sin_mean = torch.sin(ps).mean(dim=(1, 3))   # [B, n_patches]
            cos_mean = torch.cos(ps).mean(dim=(1, 3))   # [B, n_patches]
            phase = torch.atan2(sin_mean, cos_mean)
        return phase

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected 3D input [B, C, T], got shape {tuple(x.shape)}")
        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} channels, got {x.shape[1]}"
            )

        phase = self._phase_per_token(x)         # [B, n_patches]

        h = self.patch_embed(x)                  # [B, d_model, n_patches]
        h = h.transpose(1, 2)                    # [B, n_patches, d_model]
        h = h + self.pos_embed
        h = self.input_dropout(h)

        for layer in self.layers:
            h = layer(h, phase)
        h = self.norm(h)
        h = h.mean(dim=1)
        return self.classifier(h)

    def lambda_summary(self) -> dict[str, list[float]]:
        """Return per-layer per-head lambda values for logging/inspection."""
        return {
            f"layer_{i}": layer.attn.lam.detach().cpu().tolist()
            for i, layer in enumerate(self.layers)
        }
