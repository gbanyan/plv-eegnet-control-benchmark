"""EEGNet baseline for motor imagery decoding.

Reference: Lawhence et al., 2018 (https://arxiv.org/abs/1611.08024).
Input shape: [batch, channels, time].
"""

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        n_samples: int = 1000,
        sampling_rate: int = 250,
        temporal_kernel: int | None = None,
        F1: int = 8,
        D: int = 2,
        F2: int | None = None,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Default temporal kernel covers ~half a second.
        if temporal_kernel is None:
            temporal_kernel = sampling_rate // 2
        if F2 is None:
            F2 = F1 * D

        self.n_channels = n_channels
        self.n_samples = n_samples

        # Block 1: temporal conv -> depthwise spatial conv -> pool.
        self.temporal_conv = nn.Conv2d(
            1, F1, kernel_size=(1, temporal_kernel),
            padding=(0, temporal_kernel // 2), bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise_conv = nn.Conv2d(
            F1, F1 * D, kernel_size=(n_channels, 1),
            groups=F1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: separable conv -> pool.
        self.separable_depth = nn.Conv2d(
            F1 * D, F1 * D, kernel_size=(1, 16),
            padding=(0, 8), groups=F1 * D, bias=False,
        )
        self.separable_point = nn.Conv2d(F1 * D, F2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        self.elu = nn.ELU()

        # Determine flattened size with a dummy pass.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            feat = self._forward_features(dummy)
            flat = feat.view(1, -1).shape[1]

        self.classifier = nn.Linear(flat, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, 1, C, T]
        if x.dim() != 3:
            raise ValueError(f"expected 3D input [B, C, T], got shape {tuple(x.shape)}")
        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} channels, got {x.shape[1]}"
            )

        x = x.unsqueeze(1)
        x = self._forward_features(x)
        x = x.flatten(1)
        return self.classifier(x)
