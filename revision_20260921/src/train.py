"""Training entry point for the EEGNet baseline.

Run from the repo root:
    python -m src.train --config configs/eegnet_bci2a.yaml
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.datasets.bci2a import BCI2aNPZ
from src.models.eegnet import EEGNet


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, split: str) -> BCI2aNPZ:
    return BCI2aNPZ(
        npz_path=cfg["dataset"][f"{split}_path"],
        sampling_rate=cfg["dataset"]["sampling_rate"],
        bandpass=tuple(cfg["dataset"]["bandpass"]) if cfg["dataset"].get("bandpass") else None,
        standardize=cfg["dataset"].get("standardize", True),
    )


@torch.no_grad()
def run_validation(model, loader, device, criterion) -> tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        x, y = batch[:2]
        trial_ids = batch[2] if len(batch) == 3 else None
        x, y = x.to(device), y.to(device)
        if trial_ids is not None:
            trial_ids = trial_ids.to(device)
        logits = model(x, trial_ids=trial_ids) if trial_ids is not None else model(x)
        loss_sum += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = _pick_device()
    print(f"[train] device: {device}")

    train_set = build_dataset(cfg, "train")
    val_set = build_dataset(cfg, "val")
    print(f"[train] train trials: {len(train_set)}  val trials: {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False)

    model_cfg = cfg["model"]
    model = EEGNet(
        n_channels=model_cfg.get("n_channels", train_set.n_channels),
        n_classes=model_cfg.get("n_classes", max(train_set.n_classes, 4)),
        n_samples=model_cfg.get("n_samples", train_set.n_samples),
        sampling_rate=cfg["dataset"]["sampling_rate"],
        temporal_kernel=model_cfg.get("temporal_kernel"),
        F1=model_cfg.get("F1", 8),
        D=model_cfg.get("D", 2),
        F2=model_cfg.get("F2"),
        dropout=model_cfg.get("dropout", 0.5),
    ).to(device)

    optim_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg["lr"],
        weight_decay=optim_cfg.get("weight_decay", 1e-4),
    )
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = Path(cfg.get("checkpoint_dir", "results/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    epochs = cfg["epochs"]
    for epoch in range(1, epochs + 1):
        model.train()
        total, loss_sum = 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            total += x.size(0)
        train_loss = loss_sum / max(total, 1)
        val_loss, val_acc = run_validation(model, val_loader, device, criterion)

        print(
            f"epoch {epoch:03d}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model": model.state_dict(), "config": cfg, "val_acc": val_acc},
                ckpt_dir / "best.pt",
            )

    torch.save(
        {"model": model.state_dict(), "config": cfg, "val_acc": val_acc},
        ckpt_dir / "last.pt",
    )
    print(f"[train] best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
