"""Evaluate a trained EEGNet checkpoint on a .npz split.

Run from the repo root:
    python -m src.eval --config configs/eegnet_bci2a.yaml --checkpoint results/checkpoints/best.pt
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.bci2a import BCI2aNPZ
from src.models.eegnet import EEGNet
from src.train import build_dataset, load_config, run_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    from src.train import _pick_device
    device = _pick_device()

    dataset: BCI2aNPZ = build_dataset(cfg, args.split)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False)

    model_cfg = cfg["model"]
    model = EEGNet(
        n_channels=model_cfg.get("n_channels", dataset.n_channels),
        n_classes=model_cfg.get("n_classes", max(dataset.n_classes, 4)),
        n_samples=model_cfg.get("n_samples", dataset.n_samples),
        sampling_rate=cfg["dataset"]["sampling_rate"],
        temporal_kernel=model_cfg.get("temporal_kernel"),
        F1=model_cfg.get("F1", 8),
        D=model_cfg.get("D", 2),
        F2=model_cfg.get("F2"),
        dropout=model_cfg.get("dropout", 0.5),
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])

    loss, acc = run_validation(model, loader, device, nn.CrossEntropyLoss())
    print(f"[eval] split={args.split}  loss={loss:.4f}  acc={acc:.4f}")


if __name__ == "__main__":
    main()
