"""Tests for position–move dataset construction."""

import chess
import pandas as pd
import pytest

from yeafins.data.encode import decode_move
from yeafins.data.positions import (
    PositionBuildError,
    build_game_positions,
    build_position_dataset,
    make_sample_id,
    player_turn_matches,
)


def make_game_row(
    *,
    game_id: str = "game-1",
    player_color: str = "white",
    moves_uci: list[str] | None = None,
    split: str = "train",
) -> pd.Series:
    """Create a minimal valid game-level record."""
    if moves_uci is None:
        moves_uci = [
            "e2e4",
            "e7e5",
            "g1f3",
            "b8c6",
            "f1b5",
            "a7a6",
        ]

    return pd.Series(
        {
            "game_id": game_id,
            "split": split,
            "player_color": player_color,
            "moves_uci": moves_uci,
            "time_control": "600",
            "time_class": "rapid",
            "player_rating": 1200,
            "opponent_rating": 1250,
            "player_result": "win",
            "eco": "C60",
            "opening": "Ruy Lopez",
            "date": "2026.07.16",
            "source_file": "2026-07.pgn",
        }
    )


def test_make_sample_id_is_deterministic() -> None:
    assert make_sample_id("game-1", 4) == make_sample_id("game-1", 4)
    assert make_sample_id("game-1", 4) != make_sample_id("game-1", 5)


def test_player_turn_matches() -> None:
    board = chess.Board()

    assert player_turn_matches(board, "white")
    assert not player_turn_matches(board, "black")

    board.push_uci("e2e4")

    assert not player_turn_matches(board, "white")
    assert player_turn_matches(board, "black")


def test_build_game_positions_as_white() -> None:
    records = build_game_positions(make_game_row(player_color="white"))

    assert len(records) == 3
    assert [record["move_uci"] for record in records] == [
        "e2e4",
        "g1f3",
        "f1b5",
    ]
    assert [record["ply_index"] for record in records] == [0, 2, 4]


def test_build_game_positions_as_black() -> None:
    records = build_game_positions(make_game_row(player_color="black"))

    assert len(records) == 3
    assert [record["move_uci"] for record in records] == [
        "e7e5",
        "b8c6",
        "a7a6",
    ]
    assert [record["ply_index"] for record in records] == [1, 3, 5]


def test_target_labels_decode_to_original_moves() -> None:
    records = build_game_positions(make_game_row(player_color="white"))

    for record in records:
        board = chess.Board(record["fen"])
        decoded = decode_move(int(record["move_label"]), board)

        assert decoded.uci() == record["move_uci"]


def test_build_position_dataset_preserves_game_splits() -> None:
    games = pd.DataFrame(
        [
            make_game_row(
                game_id="train-game",
                split="train",
            ),
            make_game_row(
                game_id="test-game",
                split="test",
                player_color="black",
            ),
        ]
    )

    positions = build_position_dataset(games)

    assert set(
        positions.loc[
            positions["game_id"] == "train-game",
            "split",
        ]
    ) == {"train"}

    assert set(
        positions.loc[
            positions["game_id"] == "test-game",
            "split",
        ]
    ) == {"test"}


def test_build_game_positions_rejects_illegal_move() -> None:
    row = make_game_row(
        moves_uci=["e2e5"],
    )

    with pytest.raises(PositionBuildError, match="Illegal move"):
        build_game_positions(row)


def test_build_position_dataset_rejects_missing_columns() -> None:
    games = pd.DataFrame([{"game_id": "game-1"}])

    with pytest.raises(PositionBuildError, match="missing required columns"):
        build_position_dataset(games)
