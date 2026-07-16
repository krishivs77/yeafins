"""Tests for the PyTorch chess policy dataset."""

from pathlib import Path

import pandas as pd
import pytest
import torch

from yeafins.data.board import BOARD_SHAPE
from yeafins.training.dataset import (
    ChessPolicyDataset,
    create_dataloader,
)


def write_test_positions(path: Path) -> None:
    """Write a small position-level Parquet fixture."""
    dataframe = pd.DataFrame(
        [
            {
                "sample_id": "train-1",
                "game_id": "game-1",
                "split": "train",
                "fen": ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
                "move_label": 877,
                "move_uci": "e2e4",
                "player_rating": 1000,
                "opponent_rating": 1050,
                "time_class": "rapid",
            },
            {
                "sample_id": "train-2",
                "game_id": "game-2",
                "split": "train",
                "fen": ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
                "move_label": 4584,
                "move_uci": "g1f3",
                "player_rating": 1000,
                "opponent_rating": 1050,
                "time_class": "rapid",
            },
            {
                "sample_id": "val-1",
                "game_id": "game-3",
                "split": "val",
                "fen": ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
                "move_label": 877,
                "move_uci": "e2e4",
                "player_rating": None,
                "opponent_rating": None,
                "time_class": "blitz",
            },
        ]
    )

    dataframe.to_parquet(path, index=False)


def test_dataset_filters_requested_split(tmp_path: Path) -> None:
    positions_path = tmp_path / "positions.parquet"
    write_test_positions(positions_path)

    train_dataset = ChessPolicyDataset(
        positions_path=positions_path,
        split="train",
    )
    val_dataset = ChessPolicyDataset(
        positions_path=positions_path,
        split="val",
    )

    assert len(train_dataset) == 2
    assert len(val_dataset) == 1


def test_dataset_sample_shapes_and_types(tmp_path: Path) -> None:
    positions_path = tmp_path / "positions.parquet"
    write_test_positions(positions_path)

    dataset = ChessPolicyDataset(
        positions_path=positions_path,
        split="train",
    )

    sample = dataset[0]

    assert sample["board"].shape == BOARD_SHAPE
    assert sample["board"].dtype == torch.float32
    assert sample["target"].shape == ()
    assert sample["target"].dtype == torch.long
    assert sample["sample_id"] == "train-1"
    assert sample["game_id"] == "game-1"


def test_dataset_preserves_optional_metadata(tmp_path: Path) -> None:
    positions_path = tmp_path / "positions.parquet"
    write_test_positions(positions_path)

    train_dataset = ChessPolicyDataset(
        positions_path=positions_path,
        split="train",
    )
    val_dataset = ChessPolicyDataset(
        positions_path=positions_path,
        split="val",
    )

    assert train_dataset[0]["player_rating"].item() == 1000.0
    assert torch.isnan(val_dataset[0]["player_rating"])


def test_dataloader_batches_samples(tmp_path: Path) -> None:
    positions_path = tmp_path / "positions.parquet"
    write_test_positions(positions_path)

    loader = create_dataloader(
        positions_path=positions_path,
        split="train",
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(loader))

    assert batch["board"].shape == (2, *BOARD_SHAPE)
    assert batch["target"].shape == (2,)
    assert batch["target"].dtype == torch.long
    assert batch["sample_id"] == ["train-1", "train-2"]


def test_dataset_rejects_invalid_split(tmp_path: Path) -> None:
    positions_path = tmp_path / "positions.parquet"
    write_test_positions(positions_path)

    with pytest.raises(ValueError, match="split"):
        ChessPolicyDataset(
            positions_path=positions_path,
            split="development",
        )


def test_dataset_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ChessPolicyDataset(
            positions_path=tmp_path / "missing.parquet",
            split="train",
        )


def test_dataloader_rejects_invalid_batch_size(tmp_path: Path) -> None:
    positions_path = tmp_path / "positions.parquet"
    write_test_positions(positions_path)

    with pytest.raises(ValueError, match="batch_size"):
        create_dataloader(
            positions_path=positions_path,
            split="train",
            batch_size=0,
        )
