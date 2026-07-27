"""Create game-level win-only datasets for policy-model experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_GAMES_PATH = Path("data/processed/games_split.parquet")
DEFAULT_POSITIONS_PATH = Path("data/processed/positions.parquet")
DEFAULT_OUTPUT_GAMES_PATH = Path("data/processed/games_wins_only.parquet")
DEFAULT_OUTPUT_POSITIONS_PATH = Path("data/processed/positions_wins_only.parquet")
DEFAULT_SUMMARY_PATH = Path("data/processed/wins_only_summary.json")


def filter_winning_games(
    games: pd.DataFrame,
    positions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain only positions belonging to games the player won."""
    required_game_columns = {
        "game_id",
        "player_result",
        "split",
    }
    required_position_columns = {
        "game_id",
        "player_result",
        "split",
    }

    missing_games = required_game_columns - set(games.columns)
    missing_positions = required_position_columns - set(positions.columns)

    if missing_games:
        raise ValueError(
            "Games table is missing required columns: " + ", ".join(sorted(missing_games))
        )

    if missing_positions:
        raise ValueError(
            "Positions table is missing required columns: " + ", ".join(sorted(missing_positions))
        )

    winning_games = games.loc[games["player_result"] == "win"].copy()

    winning_game_ids = set(winning_games["game_id"].astype(str))

    winning_positions = positions.loc[
        positions["game_id"].astype(str).isin(winning_game_ids)
    ].copy()

    if not (winning_positions["player_result"] == "win").all():
        raise ValueError("Filtered positions contain non-winning games")

    return (
        winning_games.reset_index(drop=True),
        winning_positions.reset_index(drop=True),
    )


def build_summary(
    games: pd.DataFrame,
    positions: pd.DataFrame,
) -> dict[str, object]:
    """Summarize the filtered dataset."""
    games_by_split = games["split"].value_counts().sort_index().to_dict()
    positions_by_split = positions["split"].value_counts().sort_index().to_dict()

    return {
        "games": int(len(games)),
        "positions": int(len(positions)),
        "games_by_split": {str(key): int(value) for key, value in games_by_split.items()},
        "positions_by_split": {str(key): int(value) for key, value in positions_by_split.items()},
        "unique_game_ids": int(positions["game_id"].nunique()),
        "player_results": (positions["player_result"].value_counts().to_dict()),
    }


def create_win_only_dataset(
    *,
    games_path: Path,
    positions_path: Path,
    output_games_path: Path,
    output_positions_path: Path,
    summary_path: Path,
) -> dict[str, object]:
    """Create and save win-only game and position tables."""
    games = pd.read_parquet(games_path)
    positions = pd.read_parquet(positions_path)

    winning_games, winning_positions = filter_winning_games(
        games,
        positions,
    )

    output_games_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_positions_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    winning_games.to_parquet(
        output_games_path,
        index=False,
    )
    winning_positions.to_parquet(
        output_positions_path,
        index=False,
    )

    summary = build_summary(
        winning_games,
        winning_positions,
    )

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=("Create game-level win-only Yeafins datasets."))
    parser.add_argument(
        "--games",
        type=Path,
        default=DEFAULT_GAMES_PATH,
    )
    parser.add_argument(
        "--positions",
        type=Path,
        default=DEFAULT_POSITIONS_PATH,
    )
    parser.add_argument(
        "--output-games",
        type=Path,
        default=DEFAULT_OUTPUT_GAMES_PATH,
    )
    parser.add_argument(
        "--output-positions",
        type=Path,
        default=DEFAULT_OUTPUT_POSITIONS_PATH,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
    )
    return parser.parse_args()


def main() -> None:
    """Run win-only filtering."""
    args = parse_args()

    summary = create_win_only_dataset(
        games_path=args.games,
        positions_path=args.positions,
        output_games_path=args.output_games,
        output_positions_path=args.output_positions,
        summary_path=args.summary,
    )

    print("Created win-only dataset")
    print(f"Games:     {summary['games']}")
    print(f"Positions: {summary['positions']}")

    print("\nGames by split")
    for split, count in summary["games_by_split"].items():
        print(f"  {split}: {count}")

    print("\nPositions by split")
    for split, count in summary["positions_by_split"].items():
        print(f"  {split}: {count}")


if __name__ == "__main__":
    main()
