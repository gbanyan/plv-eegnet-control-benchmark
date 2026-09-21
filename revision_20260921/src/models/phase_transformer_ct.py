"""Phase-biased Transformer with channel×time tokenization.

Phase 2c follow-up #1. Hypothesis: the previous PhaseTransformer collapses
22 channels into a single d_model-dim token via patch_embed, then
represents phase as one scalar per token (circular mean across channels
*and* patch samples). That averaging destroys cross-channel phase
structure — which is arguably where the motor-imagery signal actually
lives.

This model keeps each (channel, patch) as its own token:

    n_tokens = n_channels * n_patches   (22 * 20 = 440 at patch_size=50)

Phase per token is per-channel circular mean over the patch samples only,
so the attention bias sees genuine cross-channel coherence:

    score_ij = (Q_i K_j^T) / sqrt(d) + lambda_h * cos(theta_i - theta_j)

where token i comes from channel c_i at patch p_i, token j from channel
c_j at patch p_j, and theta is per-channel-per-patch instantaneous phase.

Reuses PhaseBiasedAttention and PhaseEncoderLayer from phase_transformer.

Input shape: [batch, channels, time].
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.phase_transformer import PhaseEncoderLayer
from src.preprocessing.filters import instantaneous_phase_torch


class PhaseTransformerCT(nn.Module):
    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        n_samples: int = 1000,
        patch_size: int = 50,
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
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.n_channels = n_channels
        self.patch_size = patch_size
        self.sampling_rate = sampling_rate
        self.phase_band = phase_band
        self.n_patches = n_samples // patch_size
        self.n_tokens = n_channels * self.n_patches

        # ViT-style patch embedding: shared 1D conv applied to each channel
        # independently. Weights are shared across channels so the per-token
        # feature extractor has a strong inductive bias (treat all channels
        # the same at the patch level), while channel identity is preserved
        # via a learned channel embedding added after projection.
        self.patch_embed = nn.Conv1d(
            1, d_model, kernel_size=patch_size, stride=patch_size,
        )
        self.d_model = d_model

        # Factorized positional info: channel id + time-patch position.
        # Both broadcast and sum into the per-token features.
        self.channel_embed = nn.Parameter(
            torch.randn(n_channels, d_model) * 0.02
        )
        self.time_embed = nn.Parameter(
            torch.randn(self.n_patches, d_model) * 0.02
        )
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
        """Per-channel circular-mean phase over each patch.

        Returns a [B, n_tokens] tensor where token index = c * n_patches + p,
        matching the order in which patch_embed flattens its output.
        """
        low, high = (None, None) if self.phase_band is None else self.phase_band
        with torch.no_grad():
            phase_per_sample = instantaneous_phase_torch(
                x, sampling_rate=self.sampling_rate, low=low, high=high,
            )  # [B, C, T]
            B, C, T = phase_per_sample.shape
            ps = phase_per_sample.reshape(B, C, self.n_patches, self.patch_size)
            # Circular mean across patch_samples only — keeps per-channel phase.
            sin_mean = torch.sin(ps).mean(dim=-1)  # [B, C, n_patches]
            cos_mean = torch.cos(ps).mean(dim=-1)  # [B, C, n_patches]
            phase = torch.atan2(sin_mean, cos_mean)  # [B, C, n_patches]
            return phase.reshape(B, self.n_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected 3D input [B, C, T], got shape {tuple(x.shape)}")
        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} channels, got {x.shape[1]}"
            )
        B = x.shape[0]

        phase = self._phase_per_token(x)  # [B, n_tokens]

        # Apply shared patch_embed across channels by folding C into batch.
        # x: [B, C, T] -> [B*C, 1, T]
        h = x.reshape(B * self.n_channels, 1, x.shape[-1])
        h = self.patch_embed(h)  # [B*C, d_model, n_patches]
        h = h.reshape(B, self.n_channels, self.d_model, self.n_patches)
        h = h.permute(0, 1, 3, 2).contiguous()  # [B, C, n_patches, d_model]

        # Add factorized channel + time embeddings (broadcast).
        h = h + self.channel_embed.view(1, self.n_channels, 1, self.d_model)
        h = h + self.time_embed.view(1, 1, self.n_patches, self.d_model)

        # Flatten (C, n_patches) -> n_tokens, matching the phase ordering.
        h = h.reshape(B, self.n_tokens, self.d_model)
        h = self.input_dropout(h)

        for layer in self.layers:
            h = layer(h, phase)
        h = self.norm(h)
        h = h.mean(dim=1)  # mean-pool over all (channel, patch) tokens
        return self.classifier(h)

    def lambda_summary(self) -> dict[str, list[float]]:
        return {
            f"layer_{i}": layer.attn.lam.detach().cpu().tolist()
            for i, layer in enumerate(self.layers)
        }
