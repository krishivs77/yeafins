"""Expand game-level records into player position–move training examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import chess
import pandas as pd
from tqdm import tqdm

from yeafins.data.encode import encode_move

DEFAULT_INPUT_PATH = Path("data/processed/games_split.parquet")
DEFAULT_OUTPUT_PATH = Path("data/processed/positions.parquet")
DEFAULT_SUMMARY_PATH = Path("data/processed/positions_summary.json")


class PositionBuildError(RuntimeError):
    """Raised when a stored game cannot be replayed safely."""


def make_sample_id(game_id: str, ply_index: int) -> str:
    """Create a stable identifier for one position–move sample."""
    payload = f"{game_id}:{ply_index}".encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def player_turn_matches(board: chess.Board, player_color: str) -> bool:
    """Return whether the current side to move is the modeled player."""
    if player_color == "white":
        return board.turn == chess.WHITE

    if player_color == "black":
        return board.turn == chess.BLACK

    raise PositionBuildError(f"Unknown player color: {player_color!r}")


def normalize_move_list(value: Any) -> list[str]:
    """Convert a Parquet-backed sequence into a regular Python list."""
    if value is None:
        raise PositionBuildError("moves_uci is missing")

    if isinstance(value, list):
        return [str(move) for move in value]

    if isinstance(value, tuple):
        return [str(move) for move in value]

    if hasattr(value, "tolist"):
        converted = value.tolist()

        if isinstance(converted, list):
            return [str(move) for move in converted]

    raise PositionBuildError(f"Unsupported moves_uci representation: {type(value).__name__}")


def build_game_positions(game_row: pd.Series) -> list[dict[str, object]]:
    """Replay one game and extract positions immediately before player moves."""
    game_id = str(game_row["game_id"])
    player_color = str(game_row["player_color"])
    moves_uci = normalize_move_list(game_row["moves_uci"])

    board = chess.Board()
    records: list[dict[str, object]] = []

    for ply_index, move_uci in enumerate(moves_uci):
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError as exc:
            raise PositionBuildError(f"Invalid UCI move {move_uci!r} in game {game_id}") from exc

        if move not in board.legal_moves:
            raise PositionBuildError(
                f"Illegal move {move_uci} at ply {ply_index} in game {game_id}: {board.fen()}"
            )

        if player_turn_matches(board, player_color):
            records.append(
                {
                    "sample_id": make_sample_id(game_id, ply_index),
                    "game_id": game_id,
                    "split": str(game_row["split"]),
                    "ply_index": ply_index,
                    "move_number": board.fullmove_number,
                    "player_color": player_color,
                    "fen": board.fen(),
                    "move_uci": move.uci(),
                    "move_san": board.san(move),
                    "move_label": encode_move(move, board),
                    "legal_move_count": board.legal_moves.count(),
                    "time_control": str(game_row["time_control"]),
                    "time_class": str(game_row["time_class"]),
                    "player_rating": game_row["player_rating"],
                    "opponent_rating": game_row["opponent_rating"],
                    "player_result": str(game_row["player_result"]),
                    "eco": str(game_row["eco"]),
                    "opening": str(game_row["opening"]),
                    "date": str(game_row["date"]),
                    "source_file": str(game_row["source_file"]),
                }
            )

        board.push(move)

    expected_samples = (len(moves_uci) + 1) // 2 if player_color == "white" else len(moves_uci) // 2

    if len(records) != expected_samples:
        raise PositionBuildError(
            f"Expected {expected_samples} player moves in game {game_id}, "
            f"but extracted {len(records)}"
        )

    return records


def build_position_dataset(games: pd.DataFrame) -> pd.DataFrame:
    """Expand every game into player-specific position–move samples."""
    required_columns = {
        "game_id",
        "split",
        "player_color",
        "moves_uci",
        "time_control",
        "time_class",
        "player_rating",
        "opponent_rating",
        "player_result",
        "eco",
        "opening",
        "date",
        "source_file",
    }

    missing = required_columns - set(games.columns)

    if missing:
        raise PositionBuildError(
            "Game dataset is missing required columns: " + ", ".join(sorted(missing))
        )

    all_records: list[dict[str, object]] = []

    for _, game_row in tqdm(
        games.iterrows(),
        total=len(games),
        desc="Building position samples",
    ):
        all_records.extend(build_game_positions(game_row))

    positions = pd.DataFrame.from_records(all_records)

    if positions.empty:
        raise PositionBuildError("No position samples were generated")

    if positions["sample_id"].duplicated().any():
        raise PositionBuildError("Duplicate sample IDs were generated")

    return positions.sort_values(["split", "game_id", "ply_index"]).reset_index(drop=True)


def build_summary(positions: pd.DataFrame) -> dict[str, object]:
    """Create an audit summary for the position-level dataset."""
    split_counts = positions["split"].value_counts().sort_index()
    time_class_counts = positions.groupby(["split", "time_class"]).size().unstack(fill_value=0)

    return {
        "total_positions": int(len(positions)),
        "unique_games": int(positions["game_id"].nunique()),
        "unique_samples": int(positions["sample_id"].nunique()),
        "move_label_minimum": int(positions["move_label"].min()),
        "move_label_maximum": int(positions["move_label"].max()),
        "mean_legal_move_count": round(
            float(positions["legal_move_count"].mean()),
            3,
        ),
        "split_counts": {split: int(count) for split, count in split_counts.items()},
        "positions_by_time_class": {
            split: {time_class: int(count) for time_class, count in row.items()}
            for split, row in time_class_counts.to_dict(orient="index").items()
        },
    }


def write_dataset(positions: pd.DataFrame, output_path: Path) -> None:
    """Write position samples to Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positions.to_parquet(output_path, index=False)


def write_summary(summary: dict[str, object], output_path: Path) -> None:
    """Write the position summary as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Expand split games into position–move training examples."
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
    return parser.parse_args()


def main() -> None:
    """Run the position-dataset builder."""
    args = parse_args()

    games = pd.read_parquet(args.input)
    positions = build_position_dataset(games)
    summary = build_summary(positions)

    write_dataset(positions, args.output)
    write_summary(summary, args.summary)

    print()
    print(f"Total positions: {summary['total_positions']}")
    print(f"Unique games:    {summary['unique_games']}")
    print(f"Train samples:   {summary['split_counts'].get('train', 0)}")
    print(f"Val samples:     {summary['split_counts'].get('val', 0)}")
    print(f"Test samples:    {summary['split_counts'].get('test', 0)}")
    print(f"Dataset:         {args.output}")
    print(f"Summary:         {args.summary}")


if __name__ == "__main__":
    main()
