"""Tests for creating win-only chess datasets."""

import pandas as pd

from yeafins.data.filter_wins import (
    build_summary,
    filter_winning_games,
)


def make_games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "win-train",
                "split": "train",
                "player_result": "win",
            },
            {
                "game_id": "loss-train",
                "split": "train",
                "player_result": "loss",
            },
            {
                "game_id": "win-val",
                "split": "val",
                "player_result": "win",
            },
        ]
    )


def make_positions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "p1",
                "game_id": "win-train",
                "split": "train",
                "player_result": "win",
            },
            {
                "sample_id": "p2",
                "game_id": "win-train",
                "split": "train",
                "player_result": "win",
            },
            {
                "sample_id": "p3",
                "game_id": "loss-train",
                "split": "train",
                "player_result": "loss",
            },
            {
                "sample_id": "p4",
                "game_id": "win-val",
                "split": "val",
                "player_result": "win",
            },
        ]
    )


def test_filter_winning_games() -> None:
    games, positions = filter_winning_games(
        make_games(),
        make_positions(),
    )

    assert games["game_id"].tolist() == [
        "win-train",
        "win-val",
    ]
    assert positions["sample_id"].tolist() == [
        "p1",
        "p2",
        "p4",
    ]
    assert set(positions["player_result"]) == {"win"}


def test_filter_preserves_existing_splits() -> None:
    games, positions = filter_winning_games(
        make_games(),
        make_positions(),
    )

    assert games.set_index("game_id").loc["win-train", "split"] == "train"
    assert games.set_index("game_id").loc["win-val", "split"] == "val"

    assert set(positions["split"]) == {
        "train",
        "val",
    }


def test_build_summary() -> None:
    games, positions = filter_winning_games(
        make_games(),
        make_positions(),
    )

    summary = build_summary(games, positions)

    assert summary["games"] == 2
    assert summary["positions"] == 3
    assert summary["unique_game_ids"] == 2
    assert summary["games_by_split"] == {
        "train": 1,
        "val": 1,
    }
    assert summary["positions_by_split"] == {
        "train": 2,
        "val": 1,
    }
