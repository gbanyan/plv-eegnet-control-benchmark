"""Components of the locked September 2026 DSP revision protocol.

Training and evaluation are separate commands. This module never opens an E
partition itself. Feature caches contain deterministic transformations only;
normalization statistics and donor pools are fitted on inner-training rows.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, hilbert, sosfiltfilt
from torch import nn

from src.models.plv_eegnet import PLVEEGNet
from src.preprocessing.filters import instantaneous_phase_torch


def stream_seed(*parts) -> int:
    message = json.dumps(parts, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "little") % (2**31)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@torch.no_grad()
def plv_features(x: torch.Tensor, band=(8.0, 30.0), *, edge=False) -> torch.Tensor:
    """Same phasor/einsum PLV as the submission, optionally edge-handled.

    Edge sensitivity: reflect-pad 250 samples, fourth-order SOS Butterworth
    bandpass (zero phase), Hilbert, remove padding, crop 125 samples per edge.
    This changes only the auxiliary feature, not the backbone's EEG input.
    """
    if edge:
        raw = x.detach().cpu().numpy()
        padded = np.pad(raw, ((0, 0), (0, 0), (250, 250)), mode="reflect")
        sos = butter(4, band, btype="bandpass", fs=250, output="sos")
        filtered = sosfiltfilt(sos, padded, axis=-1)
        phase = np.angle(hilbert(filtered, axis=-1))[..., 375:-375].copy()
        phase = torch.as_tensor(phase, dtype=x.dtype, device=x.device)
    else:
        phase = instantaneous_phase_torch(x, 250, low=band[0], high=band[1])
    c, s = torch.cos(phase), torch.sin(phase)
    count = phase.shape[-1]
    real = (torch.einsum("bct,bdt->bcd", c, c) / count
            + torch.einsum("bct,bdt->bcd", s, s) / count)
    imaginary = (torch.einsum("bct,bdt->bcd", s, c) / count
                 - torch.einsum("bct,bdt->bcd", c, s) / count)
    matrix = torch.sqrt(real.square() + imaginary.square())
    upper = torch.triu_indices(x.shape[1], x.shape[1], offset=1, device=x.device)
    return matrix[:, upper[0], upper[1]]


class FrozenStandardization(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("variance", torch.ones(size))
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        return (x - self.mean) * torch.rsqrt(self.variance + 1e-5) * self.weight + self.bias


class RevisionModel(PLVEEGNet):
    """Original EEG backbone with externally cached auxiliary features."""
    def __init__(self, spec: dict, condition: dict, init_seed: int):
        torch.manual_seed(init_seed)
        normalization = condition["normalization"]
        parent_norm = normalization if normalization in ("batchnorm", "layernorm", "none") else "batchnorm"
        super().__init__(n_channels=spec["n_channels"], n_classes=spec["n_classes"],
                         n_samples=spec["n_samples"], plv_normalization=parent_norm)
        full_size = self.n_pairs
        size = condition.get("dimension", full_size)
        self.baseline = condition["mode"] == "baseline"
        if self.baseline:
            size = 0
        if size != full_size:
            if normalization == "batchnorm" and size:
                self.plv_norm = nn.BatchNorm1d(size)
            elif normalization == "layernorm" and size:
                self.plv_norm = nn.LayerNorm(size)
            else:
                self.plv_norm = nn.Identity()
            # Shared backbone initialization; dimension-specific head initialization
            # with a separate, reproducible stream. Same dimensions pair exactly.
            torch.manual_seed(stream_seed(init_seed, "reduced_head", size))
            self.classifier = nn.Linear(self.classifier.in_features - full_size + size,
                                        spec["n_classes"])
        if normalization == "frozen":
            self.plv_norm = FrozenStandardization(size)
        elif normalization == "bn_noaffine":
            self.plv_norm = nn.BatchNorm1d(size, affine=False)
        self.n_pairs = size

    def forward(self, x, features):
        embedding = self._eegnet_features(x)
        return self.from_embedding(embedding, features)

    def from_embedding(self, embedding, features):
        if self.baseline:
            return self.classifier(embedding)
        return self.classifier(torch.cat((embedding, self.plv_norm(features)), dim=1))


class FeatureControl:
    """Training-only control state, with deterministic evaluation assignments.

    The first training presentation uses the common fixed family in all refresh
    schedules. Subsequent visits redraw with probability q, independently of
    optimizer/dropout/order RNG. Thus q=1 is IID uniform within each trial.
    """
    def __init__(self, values, ids, train_indices, condition, seeds, *, donor_values=None,
                 donor_ids=None):
        self.values, self.ids = values, ids
        self.train_indices = train_indices
        self.condition, self.seeds = condition, seeds
        self.device = values.device
        self.generator = torch.Generator(device="cpu").manual_seed(seeds["permutation"])
        self.refresh_generator = torch.Generator(device="cpu").manual_seed(seeds["refresh"])
        self.donor_generator = torch.Generator(device="cpu").manual_seed(seeds["donor"])
        self.donor_values = values[train_indices] if donor_values is None else donor_values
        self.donor_ids = ids[train_indices] if donor_ids is None else donor_ids
        self._donor_position = {int(i): p for p, i in enumerate(self.donor_ids.tolist())}
        self.fixed = self.fixed_values()
        self.current = self.fixed.clone()
        self.redraw_count = 0

    def fixed_permutations(self, ids):
        p = self.values.shape[1]
        return torch.stack([
            torch.randperm(p, generator=torch.Generator().manual_seed(
                stream_seed(self.seeds["fixed"], int(i)))) for i in ids.tolist()
        ]).to(self.device)

    def donor_indices(self, ids, *, generator=None):
        size = len(self.donor_ids)
        if size < 2:
            raise ValueError("donor control requires at least two inner-training trials")
        out = []
        for identifier in ids.tolist():
            own_position = self._donor_position.get(int(identifier))
            local = generator or torch.Generator().manual_seed(
                stream_seed(self.seeds["donor_fixed"], int(identifier)))
            draw = int(torch.randint(size - (own_position is not None), (1,), generator=local))
            if own_position is not None and draw >= own_position:
                draw += 1
            out.append(draw)
        return torch.tensor(out, device=self.device, dtype=torch.long)

    def fixed_values(self):
        mode = self.condition["mode"]
        if mode in ("real", "baseline"):
            return self.values
        if mode == "zero":
            return torch.zeros_like(self.values)
        if mode == "global":
            order = torch.randperm(self.values.shape[1], generator=torch.Generator().manual_seed(
                self.seeds["global"])).to(self.device)
            return self.values[:, order]
        if mode.startswith("donor"):
            return self.donor_values[self.donor_indices(self.ids)]
        return self.values.gather(1, self.fixed_permutations(self.ids))

    def begin_epoch(self, epoch: int):
        mode = self.condition["mode"]
        if epoch == 1 or mode not in ("resampled", "refresh", "donor_resampled"):
            return
        indices = self.train_indices
        if mode == "donor_resampled":
            selected = self.donor_indices(self.ids[indices.cpu()], generator=self.donor_generator)
            self.current[indices] = self.donor_values[selected]
            self.redraw_count += len(indices)
            return
        q = self.condition.get("refresh_q", 1.0)
        if q < 1:
            retain = torch.rand(len(indices), generator=self.refresh_generator) < q
            indices = indices[retain.to(indices.device)]
        if len(indices):
            orders = torch.stack([torch.randperm(self.values.shape[1], generator=self.generator)
                                  for _ in range(len(indices))]).to(self.device)
            self.current[indices] = self.values[indices].gather(1, orders)
            self.redraw_count += len(indices)

    def calibration(self):
        """Exact empirical fixed or uniform-orbit moments using training only."""
        if self.condition["mode"] == "resampled":
            training = self.values[self.train_indices].double()
            mean = training.mean().expand(training.shape[1])
            variance = (training.square().mean() - mean.square()).clamp_min(0)
        else:
            training = self.fixed[self.train_indices].double()
            mean, variance = training.mean(0), training.var(0, unbiased=False)
        return mean.float(), variance.float()


class CheckpointSelector:
    """Strict best loss; thresholded improvement only controls patience.

    A virtual early-stop boundary is irrevocable. Training continues only until
    the predeclared fixed-budget checkpoint also exists, up to a hard maximum.
    """
    def __init__(self, *, minimum=100, patience=75, min_delta=1e-4):
        self.minimum, self.patience, self.min_delta = minimum, patience, min_delta
        self.best_loss = float("inf")
        self.patience_reference = float("inf")
        self.last_improvement = 0
        self.best_epoch = 0
        self.stop_epoch = None
        self.best_accuracy = -1.0
        self.accuracy_epoch = 0

    def update(self, epoch, loss, accuracy):
        if not np.isfinite(loss) or not np.isfinite(accuracy):
            raise ValueError("nonfinite validation statistics")
        save_loss = False
        if self.stop_epoch is None:
            if loss < self.best_loss:
                self.best_loss, self.best_epoch, save_loss = loss, epoch, True
            if loss < self.patience_reference - self.min_delta:
                self.patience_reference, self.last_improvement = loss, epoch
            if epoch >= self.minimum and epoch - self.last_improvement >= self.patience:
                self.stop_epoch = epoch
        save_accuracy = accuracy > self.best_accuracy
        if save_accuracy:
            self.best_accuracy, self.accuracy_epoch = accuracy, epoch
        return save_loss, save_accuracy

    def finished(self, epoch, fixed_budget, maximum):
        return epoch >= maximum or (self.stop_epoch is not None
                                     and epoch >= max(self.stop_epoch, fixed_budget))
