"""Train the Yeafins residual chess policy model."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from yeafins.models.resnet_policy import (
    ResNetPolicy,
    ResNetPolicyConfig,
    count_trainable_parameters,
)
from yeafins.training.dataset import create_dataloader

DEFAULT_POSITIONS_PATH = Path("data/processed/positions.parquet")
DEFAULT_RUN_DIR = Path("runs/resnet_policy")


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for policy-model training."""

    positions_path: str = str(DEFAULT_POSITIONS_PATH)
    run_dir: str = str(DEFAULT_RUN_DIR)

    seed: int = 42
    epochs: int = 30
    batch_size: int = 256
    num_workers: int = 0

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0

    trunk_channels: int = 64
    residual_blocks: int = 6

    early_stopping_patience: int = 7
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    minimum_learning_rate: float = 1e-6

    def validate(self) -> None:
        """Validate training hyperparameters."""
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")

        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")

        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")


@dataclass
class EpochMetrics:
    """Aggregated metrics for one dataset pass."""

    loss: float
    top1: float
    top3: float
    top5: float
    samples: int
    seconds: float


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Select the best available PyTorch device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def topk_correct_counts(
    logits: Tensor,
    targets: Tensor,
    topk: tuple[int, ...] = (1, 3, 5),
) -> dict[int, int]:
    """Count correct predictions for each requested top-k threshold."""
    maximum_k = max(topk)

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")

    if targets.ndim != 1:
        raise ValueError("targets must have shape [batch]")

    predictions = logits.topk(
        maximum_k,
        dim=1,
        largest=True,
        sorted=True,
    ).indices

    matches = predictions.eq(targets.unsqueeze(1))

    return {k: int(matches[:, :k].any(dim=1).sum().item()) for k in topk}


def run_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: AdamW | None = None,
    gradient_clip_norm: float = 5.0,
) -> EpochMetrics:
    """Run one training or evaluation epoch."""
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    started = time.perf_counter()

    total_loss = 0.0
    total_samples = 0
    total_correct = {1: 0, 3: 0, 5: 0}

    for batch in loader:
        boards = batch["board"].to(
            device,
            non_blocking=True,
        )
        targets = batch["target"].to(
            device,
            non_blocking=True,
        )

        batch_size = boards.shape[0]

        if training:
            optimizer.zero_grad(set_to_none=True)

            logits = model(boards)
            loss = criterion(logits, targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )

            optimizer.step()

        else:
            with torch.inference_mode():
                logits = model(boards)
                loss = criterion(logits, targets)

        correct_counts = topk_correct_counts(
            logits.detach(),
            targets,
        )

        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        for k, count in correct_counts.items():
            total_correct[k] += count

    elapsed = time.perf_counter() - started

    return EpochMetrics(
        loss=total_loss / total_samples,
        top1=total_correct[1] / total_samples,
        top3=total_correct[3] / total_samples,
        top5=total_correct[5] / total_samples,
        samples=total_samples,
        seconds=elapsed,
    )


def save_checkpoint(
    path: Path,
    *,
    model: ResNetPolicy,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_validation_loss: float,
    config: TrainingConfig,
    history: list[dict[str, object]],
) -> None:
    """Save a resumable training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_validation_loss": best_validation_loss,
            "training_config": asdict(config),
            "model_config": model.config.to_dict(),
            "history": history,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    model: ResNetPolicy,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
) -> tuple[int, float, list[dict[str, object]]]:
    """Restore a training checkpoint."""
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint["epoch"]) + 1
    best_validation_loss = float(checkpoint["best_validation_loss"])
    history = list(checkpoint.get("history", []))

    return start_epoch, best_validation_loss, history


def write_history(
    history: list[dict[str, object]],
    run_dir: Path,
) -> None:
    """Write training history to JSON and CSV."""
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "history.json"
    json_path.write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )

    if not history:
        return

    csv_path = run_dir / "history.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)


def train(config: TrainingConfig, *, resume: Path | None = None) -> None:
    """Train the residual policy network."""
    config.validate()
    set_seed(config.seed)

    device = select_device()
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader = create_dataloader(
        positions_path=Path(config.positions_path),
        split="train",
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = create_dataloader(
        positions_path=Path(config.positions_path),
        split="val",
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ResNetPolicy(
        ResNetPolicyConfig(
            trunk_channels=config.trunk_channels,
            residual_blocks=config.residual_blocks,
        )
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.minimum_learning_rate,
    )

    start_epoch = 1
    best_validation_loss = float("inf")
    history: list[dict[str, object]] = []

    if resume is not None:
        start_epoch, best_validation_loss, history = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    print(f"Device:               {device}")
    print(f"Train samples:        {len(train_loader.dataset)}")
    print(f"Validation samples:   {len(val_loader.dataset)}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"Batch size:           {config.batch_size}")
    print(f"Starting epoch:       {start_epoch}")
    print()

    epochs_without_improvement = 0

    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(val_metrics.loss)

        learning_rate = float(optimizer.param_groups[0]["lr"])

        record: dict[str, object] = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics.loss,
            "train_top1": train_metrics.top1,
            "train_top3": train_metrics.top3,
            "train_top5": train_metrics.top5,
            "train_seconds": train_metrics.seconds,
            "val_loss": val_metrics.loss,
            "val_top1": val_metrics.top1,
            "val_top3": val_metrics.top3,
            "val_top5": val_metrics.top5,
            "val_seconds": val_metrics.seconds,
        }

        history.append(record)
        write_history(history, run_dir)

        improved = val_metrics.loss < best_validation_loss

        if improved:
            best_validation_loss = val_metrics.loss
            epochs_without_improvement = 0

            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_validation_loss=best_validation_loss,
                config=config,
                history=history,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            run_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_validation_loss=best_validation_loss,
            config=config,
            history=history,
        )

        marker = " *" if improved else ""

        print(
            f"Epoch {epoch:02d} | "
            f"lr={learning_rate:.2e} | "
            f"train loss={train_metrics.loss:.4f} "
            f"top1={train_metrics.top1:.3f} "
            f"top3={train_metrics.top3:.3f} "
            f"top5={train_metrics.top5:.3f} | "
            f"val loss={val_metrics.loss:.4f} "
            f"top1={val_metrics.top1:.3f} "
            f"top3={val_metrics.top3:.3f} "
            f"top5={val_metrics.top5:.3f}"
            f"{marker}"
        )

        if epochs_without_improvement >= config.early_stopping_patience:
            print()
            print(
                "Early stopping after "
                f"{epochs_without_improvement} epochs "
                "without validation-loss improvement."
            )
            break

    config_path = run_dir / "training_config.json"
    config_path.write_text(
        json.dumps(asdict(config), indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Best checkpoint:      {run_dir / 'best.pt'}")
    print(f"Last checkpoint:      {run_dir / 'last.pt'}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train the Yeafins residual chess policy model.")

    parser.add_argument(
        "--positions",
        type=Path,
        default=DEFAULT_POSITIONS_PATH,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--trunk-channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)

    return parser.parse_args()


def main() -> None:
    """Run training from the command line."""
    args = parse_args()

    config = TrainingConfig(
        positions_path=str(args.positions),
        run_dir=str(args.run_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        trunk_channels=args.trunk_channels,
        residual_blocks=args.residual_blocks,
        seed=args.seed,
    )

    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
