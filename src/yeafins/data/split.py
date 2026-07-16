"""Create deterministic randomized game-level train, validation, and test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_INPUT_PATH = Path("data/interim/games.parquet")
DEFAULT_OUTPUT_PATH = Path("data/processed/games_split.parquet")
DEFAULT_SUMMARY_PATH = Path("data/processed/split_summary.json")

DEFAULT_SEED = 42
DEFAULT_TRAIN_FRACTION = 0.80
DEFAULT_VAL_FRACTION = 0.10
DEFAULT_TEST_FRACTION = 0.10


def rating_band(rating: int | float | None) -> str:
    """Convert a player rating into a broad stratification band."""
    if rating is None or pd.isna(rating):
        return "unknown"

    numeric_rating = int(rating)

    if numeric_rating < 600:
        return "under_600"

    if numeric_rating < 800:
        return "600_799"

    if numeric_rating < 1000:
        return "800_999"

    if numeric_rating < 1200:
        return "1000_1199"

    return "1200_plus"


def build_stratification_label(dataframe: pd.DataFrame) -> pd.Series:
    """Construct a coarse label for approximately balanced splits."""
    rating_labels = dataframe["player_rating"].map(rating_band)

    labels = (
        dataframe["time_class"].fillna("unknown").astype(str)
        + "|"
        + dataframe["player_color"].fillna("unknown").astype(str)
        + "|"
        + dataframe["player_result"].fillna("unknown").astype(str)
        + "|"
        + rating_labels
    )

    counts = labels.value_counts()

    rare_labels = counts[counts < 3].index

    return labels.where(~labels.isin(rare_labels), "rare")


def validate_split_fractions(
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> None:
    """Validate that the requested fractions form a complete split."""
    fractions = (train_fraction, val_fraction, test_fraction)

    if any(fraction <= 0 or fraction >= 1 for fraction in fractions):
        raise ValueError("All split fractions must be between 0 and 1")

    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Train, validation, and test fractions must sum to 1")


def split_games(
    dataframe: pd.DataFrame,
    *,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Assign every game to exactly one randomized split."""
    validate_split_fractions(
        train_fraction,
        val_fraction,
        test_fraction,
    )

    required_columns = {
        "game_id",
        "time_class",
        "player_color",
        "player_result",
        "player_rating",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if dataframe["game_id"].duplicated().any():
        raise ValueError("game_id values must be unique before splitting")

    working = dataframe.copy()
    working["_stratify"] = build_stratification_label(working)

    train, remainder = train_test_split(
        working,
        train_size=train_fraction,
        random_state=seed,
        shuffle=True,
        stratify=working["_stratify"],
    )

    remainder_fraction = val_fraction + test_fraction
    relative_val_fraction = val_fraction / remainder_fraction

    remainder_labels = remainder["_stratify"]
    remainder_counts = remainder_labels.value_counts()

    if (remainder_counts < 2).any():
        remainder_stratify: pd.Series | None = None
    else:
        remainder_stratify = remainder_labels

    validation, test = train_test_split(
        remainder,
        train_size=relative_val_fraction,
        random_state=seed + 1,
        shuffle=True,
        stratify=remainder_stratify,
    )

    train = train.copy()
    validation = validation.copy()
    test = test.copy()

    train["split"] = "train"
    validation["split"] = "val"
    test["split"] = "test"

    result = pd.concat(
        [train, validation, test],
        ignore_index=True,
    )

    result = result.drop(columns=["_stratify"])
    result = result.sort_values("game_id").reset_index(drop=True)

    return result


def build_split_summary(dataframe: pd.DataFrame) -> dict[str, object]:
    """Create an audit summary for the completed split."""
    split_counts = dataframe["split"].value_counts().sort_index()

    summary: dict[str, object] = {
        "total_games": int(len(dataframe)),
        "split_counts": {split: int(count) for split, count in split_counts.items()},
        "split_fractions": {
            split: round(float(count / len(dataframe)), 4) for split, count in split_counts.items()
        },
        "by_time_class": {},
        "by_player_color": {},
        "by_player_result": {},
        "by_rating_band": {},
    }

    summary["by_time_class"] = (
        dataframe.groupby(["split", "time_class"])
        .size()
        .unstack(fill_value=0)
        .astype(int)
        .to_dict(orient="index")
    )

    summary["by_player_color"] = (
        dataframe.groupby(["split", "player_color"])
        .size()
        .unstack(fill_value=0)
        .astype(int)
        .to_dict(orient="index")
    )

    summary["by_player_result"] = (
        dataframe.groupby(["split", "player_result"])
        .size()
        .unstack(fill_value=0)
        .astype(int)
        .to_dict(orient="index")
    )

    rating_labels = dataframe["player_rating"].map(rating_band)

    summary["by_rating_band"] = (
        dataframe.assign(rating_band=rating_labels)
        .groupby(["split", "rating_band"])
        .size()
        .unstack(fill_value=0)
        .astype(int)
        .to_dict(orient="index")
    )

    return summary


def write_split(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Write the split dataset to Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)


def write_summary(summary: dict[str, object], output_path: Path) -> None:
    """Write the split summary as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create randomized game-level train/validation/test splits."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    return parser.parse_args()


def main() -> None:
    """Run the dataset splitting CLI."""
    args = parse_args()

    dataframe = pd.read_parquet(args.input)

    split_dataframe = split_games(
        dataframe,
        seed=args.seed,
    )

    summary = build_split_summary(split_dataframe)

    write_split(split_dataframe, args.output)
    write_summary(summary, args.summary)

    print()
    print(f"Total games: {summary['total_games']}")

    split_counts = summary["split_counts"]

    if isinstance(split_counts, dict):
        print(f"Train games: {split_counts.get('train', 0)}")
        print(f"Val games:   {split_counts.get('val', 0)}")
        print(f"Test games:  {split_counts.get('test', 0)}")

    print(f"Split dataset: {args.output}")
    print(f"Split summary: {args.summary}")


if __name__ == "__main__":
    main()
