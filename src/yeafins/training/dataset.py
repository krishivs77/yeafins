"""PyTorch dataset for Yeafins chess position–move samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chess
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from yeafins.data.board import encode_fen
from yeafins.data.encode import POLICY_SIZE, legal_move_mask

DEFAULT_POSITIONS_PATH = Path("data/processed/positions.parquet")


class ChessPolicyDataset(Dataset[dict[str, Any]]):
    """Load encoded chess positions and policy labels from Parquet."""

    def __init__(
        self,
        positions_path: Path = DEFAULT_POSITIONS_PATH,
        *,
        split: str,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train, val, or test; received {split!r}")

        if not positions_path.exists():
            raise FileNotFoundError(f"Position dataset does not exist: {positions_path}")

        dataframe = pd.read_parquet(
            positions_path,
            columns=[
                "sample_id",
                "game_id",
                "split",
                "fen",
                "move_label",
                "move_uci",
                "player_color",
                "player_rating",
                "opponent_rating",
                "time_class",
            ],
        )

        dataframe = dataframe.loc[dataframe["split"] == split].reset_index(drop=True)

        if dataframe.empty:
            raise ValueError(f"No rows found for split {split!r} in {positions_path}")

        if not dataframe["move_label"].between(0, POLICY_SIZE - 1).all():
            raise ValueError("Dataset contains move labels outside policy space")

        self.positions_path = positions_path
        self.split = split
        self.dataframe = dataframe

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Encode and return one chess training sample."""
        row = self.dataframe.iloc[index]

        board_array = encode_fen(str(row["fen"]))
        board_tensor = torch.from_numpy(board_array)

        move_label = int(row["move_label"])

        return {
            "board": board_tensor,
            "target": torch.tensor(move_label, dtype=torch.long),
            "sample_id": str(row["sample_id"]),
            "game_id": str(row["game_id"]),
            "fen": str(row["fen"]),
            "move_uci": str(row["move_uci"]),
            "player_color": str(row["player_color"]),
            "player_rating": self._optional_float(row["player_rating"]),
            "opponent_rating": self._optional_float(row["opponent_rating"]),
            "time_class": str(row["time_class"]),
        }

    @staticmethod
    def _optional_float(value: Any) -> Tensor:
        """Convert optional numeric metadata into a float tensor."""
        if value is None or pd.isna(value):
            return torch.tensor(float("nan"), dtype=torch.float32)

        return torch.tensor(float(value), dtype=torch.float32)


def create_dataloader(
    positions_path: Path = DEFAULT_POSITIONS_PATH,
    *,
    split: str,
    batch_size: int,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader[dict[str, Any]]:
    """Create a DataLoader for one dataset split."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    dataset = ChessPolicyDataset(
        positions_path=positions_path,
        split=split,
    )

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_legal_mask_tensor(
    fens: list[str] | tuple[str, ...],
) -> Tensor:
    """Create a batched Boolean legal-move mask from FEN strings."""
    if not fens:
        raise ValueError("At least one FEN is required")

    masks: list[np.ndarray] = []

    for fen in fens:
        try:
            board = chess.Board(str(fen))
        except ValueError as exc:
            raise ValueError(f"Invalid FEN: {fen!r}") from exc

        masks.append(legal_move_mask(board))

    stacked = np.stack(masks, axis=0)
    return torch.from_numpy(stacked)
