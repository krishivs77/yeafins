"""Tests for game-level dataset splitting."""

import pandas as pd
import pytest

from yeafins.data.split import (
    build_stratification_label,
    rating_band,
    split_games,
    validate_split_fractions,
)


def make_dataset(size: int = 200) -> pd.DataFrame:
    """Create a representative synthetic game dataset."""
    rows = []

    time_classes = ["rapid", "blitz", "bullet"]
    colors = ["white", "black"]
    results = ["win", "loss", "draw"]
    ratings = [500, 700, 900, 1100, 1300]

    for index in range(size):
        rows.append(
            {
                "game_id": f"game-{index:04d}",
                "time_class": time_classes[index % len(time_classes)],
                "player_color": colors[index % len(colors)],
                "player_result": results[index % len(results)],
                "player_rating": ratings[index % len(ratings)],
            }
        )

    return pd.DataFrame(rows)


def test_rating_band() -> None:
    assert rating_band(500) == "under_600"
    assert rating_band(600) == "600_799"
    assert rating_band(799) == "600_799"
    assert rating_band(800) == "800_999"
    assert rating_band(1000) == "1000_1199"
    assert rating_band(1200) == "1200_plus"
    assert rating_band(None) == "unknown"


def test_validate_split_fractions() -> None:
    validate_split_fractions(0.8, 0.1, 0.1)

    with pytest.raises(ValueError):
        validate_split_fractions(0.8, 0.2, 0.2)

    with pytest.raises(ValueError):
        validate_split_fractions(1.0, 0.0, 0.0)


def test_build_stratification_label() -> None:
    dataframe = make_dataset(30)

    labels = build_stratification_label(dataframe)

    assert len(labels) == len(dataframe)
    assert labels.notna().all()


def test_split_games_assigns_every_game_once() -> None:
    dataframe = make_dataset()

    result = split_games(dataframe, seed=42)

    assert len(result) == len(dataframe)
    assert result["game_id"].nunique() == len(dataframe)
    assert set(result["split"]) == {"train", "val", "test"}


def test_split_games_has_expected_sizes() -> None:
    dataframe = make_dataset()

    result = split_games(dataframe, seed=42)

    counts = result["split"].value_counts()

    assert counts["train"] == 160
    assert counts["val"] == 20
    assert counts["test"] == 20


def test_split_games_is_deterministic() -> None:
    dataframe = make_dataset()

    first = split_games(dataframe, seed=42)
    second = split_games(dataframe, seed=42)

    first_assignments = first.set_index("game_id")["split"].sort_index()
    second_assignments = second.set_index("game_id")["split"].sort_index()

    pd.testing.assert_series_equal(
        first_assignments,
        second_assignments,
    )


def test_split_games_changes_with_seed() -> None:
    dataframe = make_dataset()

    first = split_games(dataframe, seed=42)
    second = split_games(dataframe, seed=99)

    first_assignments = first.set_index("game_id")["split"].sort_index()
    second_assignments = second.set_index("game_id")["split"].sort_index()

    assert not first_assignments.equals(second_assignments)


def test_split_games_rejects_duplicate_game_ids() -> None:
    dataframe = make_dataset()
    dataframe.loc[1, "game_id"] = dataframe.loc[0, "game_id"]

    with pytest.raises(ValueError, match="game_id"):
        split_games(dataframe)
