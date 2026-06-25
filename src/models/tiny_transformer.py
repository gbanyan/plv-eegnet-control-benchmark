"""Tiny Transformer baseline for motor imagery decoding.

Patch-based tokenization: a 1D conv with kernel == stride == patch_size
projects each non-overlapping time-patch (across all channels) into a
single d_model-dim token. Positional embedding is learned. Classification
uses mean-pooling over tokens.

Phase 2a of the phase-attention plan: this is the *plain* Transformer
control, with no phase bias. Use it to attribute Phase 2c gains to phase
specifically rather than to attention as a mechanism.

Input shape: [batch, channels, time].
"""

import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        n_samples: int = 1000,
        patch_size: int = 25,        # 100 ms at 250 Hz -> 40 tokens
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.5,
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
        n_patches = n_samples // patch_size

        # Patch embedding: Conv1d with kernel=stride=patch_size acts as
        # "flatten the patch and linearly project to d_model" but is a
        # cleaner/faster way to write it.
        self.patch_embed = nn.Conv1d(
            n_channels, d_model, kernel_size=patch_size, stride=patch_size,
        )
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected 3D input [B, C, T], got shape {tuple(x.shape)}")
        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} channels, got {x.shape[1]}"
            )

        h = self.patch_embed(x)        # [B, d_model, n_patches]
        h = h.transpose(1, 2)           # [B, n_patches, d_model]
        h = h + self.pos_embed
        h = self.input_dropout(h)
        h = self.encoder(h)
        h = self.norm(h)
        h = h.mean(dim=1)               # mean-pool over tokens -> [B, d_model]
        return self.classifier(h)
