"""Tests for the policy-model training utilities."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from yeafins.models.resnet_policy import (
    ResNetPolicy,
    ResNetPolicyConfig,
)
from yeafins.training.dataset import create_dataloader
from yeafins.training.train import (
    TrainingConfig,
    load_checkpoint,
    run_epoch,
    save_checkpoint,
    set_seed,
    topk_correct_counts,
)


def write_positions(path: Path) -> None:
    """Write a tiny train/validation fixture."""
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    rows = []

    for index in range(12):
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "game_id": f"game-{index}",
                "split": "train" if index < 8 else "val",
                "fen": starting_fen,
                "move_label": 877,
                "move_uci": "e2e4",
                "player_color": "white",
                "player_rating": 1000,
                "opponent_rating": 1000,
                "time_class": "rapid",
            }
        )

    pd.DataFrame(rows).to_parquet(path, index=False)


def make_model() -> ResNetPolicy:
    return ResNetPolicy(
        ResNetPolicyConfig(
            trunk_channels=8,
            residual_blocks=1,
        )
    )


def test_topk_correct_counts() -> None:
    logits = torch.tensor(
        [
            [5.0, 4.0, 3.0, 2.0],
            [1.0, 4.0, 3.0, 2.0],
        ]
    )
    targets = torch.tensor([0, 2])

    counts = topk_correct_counts(
        logits,
        targets,
        topk=(1, 2, 3),
    )

    assert counts[1] == 1
    assert counts[2] == 2
    assert counts[3] == 2


def test_training_config_validation() -> None:
    TrainingConfig().validate()

    with pytest.raises(ValueError):
        TrainingConfig(epochs=0).validate()

    with pytest.raises(ValueError):
        TrainingConfig(batch_size=0).validate()


def test_set_seed_is_deterministic() -> None:
    set_seed(42)
    first = torch.randn(4)

    set_seed(42)
    second = torch.randn(4)

    torch.testing.assert_close(first, second)


def test_run_epoch_training_and_validation(tmp_path: Path) -> None:
    positions_path = tmp_path / "positions.parquet"
    write_positions(positions_path)

    loader = create_dataloader(
        positions_path=positions_path,
        split="train",
        batch_size=4,
        shuffle=False,
    )

    model = make_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)

    training_metrics = run_epoch(
        model,
        loader,
        criterion,
        torch.device("cpu"),
        optimizer=optimizer,
    )

    validation_metrics = run_epoch(
        model,
        loader,
        criterion,
        torch.device("cpu"),
    )

    assert training_metrics.samples == 8
    assert validation_metrics.samples == 8
    assert training_metrics.loss > 0
    assert validation_metrics.loss > 0

    for metrics in (training_metrics, validation_metrics):
        assert 0 <= metrics.top1 <= 1
        assert 0 <= metrics.top3 <= 1
        assert 0 <= metrics.top5 <= 1


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = make_model()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer)

    checkpoint_path = tmp_path / "checkpoint.pt"
    config = TrainingConfig()

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        best_validation_loss=4.5,
        config=config,
        history=[{"epoch": 3}],
    )

    restored_model = make_model()
    restored_optimizer = AdamW(
        restored_model.parameters(),
        lr=1e-3,
    )
    restored_scheduler = ReduceLROnPlateau(
        restored_optimizer,
    )

    start_epoch, best_loss, history = load_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        device=torch.device("cpu"),
    )

    assert start_epoch == 4
    assert best_loss == 4.5
    assert history == [{"epoch": 3}]

    for original, restored in zip(
        model.parameters(),
        restored_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(original, restored)
