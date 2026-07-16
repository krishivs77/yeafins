"""Parse downloaded Chess.com PGNs into a clean game-level dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import chess.pgn
import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_USERNAME = "yeafins"
DEFAULT_INPUT_DIR = Path("data/raw/pgn")
DEFAULT_OUTPUT_PATH = Path("data/interim/games.parquet")
DEFAULT_SUMMARY_PATH = Path("data/interim/corpus_summary.json")

STANDARD_VARIANTS = {"", "standard", "chess"}


@dataclass(frozen=True)
class ParsedGame:
    """Serializable information for one accepted standard chess game."""

    game_id: str
    source_file: str
    source_game_index: int
    event: str
    site: str
    date: str
    utc_date: str
    utc_time: str
    end_time: str
    time_control: str
    time_class: str
    rated: bool | None
    result: str
    termination: str
    eco: str
    opening: str
    white_username: str
    black_username: str
    white_rating: int | None
    black_rating: int | None
    player_color: str
    player_rating: int | None
    opponent_rating: int | None
    player_result: str
    ply_count: int
    move_count: int
    moves_uci: list[str]
    moves_san: list[str]
    final_fen: str


@dataclass(frozen=True)
class RejectedGame:
    """Reason a PGN game was excluded from the clean corpus."""

    source_file: str
    source_game_index: int
    reason: str


def normalize_username(value: str) -> str:
    """Normalize a Chess.com username for case-insensitive comparison."""
    return value.strip().lower()


def parse_optional_int(value: str | None) -> int | None:
    """Parse an integer header while tolerating missing or invalid values."""
    if value is None:
        return None

    cleaned = value.strip()

    if not cleaned or cleaned in {"?", "-"}:
        return None

    try:
        return int(cleaned)
    except ValueError:
        return None


def parse_optional_bool(value: str | None) -> bool | None:
    """Parse common true/false PGN header representations."""
    if value is None:
        return None

    normalized = value.strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    return None


def infer_player_result(result: str, player_color: str) -> str:
    """Convert a PGN result into win, loss, draw, or unknown."""
    if result == "1/2-1/2":
        return "draw"

    if result == "1-0":
        return "win" if player_color == "white" else "loss"

    if result == "0-1":
        return "win" if player_color == "black" else "loss"

    return "unknown"


def infer_time_class(headers: chess.pgn.Headers) -> str:
    """Infer the Chess.com time class from PGN metadata."""

    for key in ("TimeClass", "TimeCategory"):
        value = headers.get(key)

        if value:
            return value.strip().lower()

    time_control = headers.get("TimeControl", "").strip()

    if not time_control:
        return "unknown"

    known_classes = {
        "30": "bullet",
        "60": "bullet",
        "60+1": "bullet",
        "120+1": "blitz",
        "180": "blitz",
        "180+2": "blitz",
        "300": "blitz",
        "300+5": "blitz",
        "600": "rapid",
        "900+10": "rapid",
        "1800": "rapid",
        "3600": "rapid",
        "1/259200": "daily",
    }

    return known_classes.get(time_control, "unknown")


def make_game_id(
    source_file: Path,
    source_game_index: int,
    headers: chess.pgn.Headers,
    moves_uci: list[str],
) -> str:
    """Create a deterministic identifier without relying on a Chess.com URL."""
    components = [
        source_file.name,
        str(source_game_index),
        headers.get("Site", ""),
        headers.get("UTCDate", headers.get("Date", "")),
        headers.get("UTCTime", ""),
        headers.get("White", ""),
        headers.get("Black", ""),
        " ".join(moves_uci),
    ]

    payload = "\n".join(components).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def read_games(pgn_path: Path) -> Iterator[tuple[int, chess.pgn.Game]]:
    """Yield each readable game from a PGN file."""
    with pgn_path.open("r", encoding="utf-8", errors="replace") as handle:
        game_index = 0

        while True:
            try:
                game = chess.pgn.read_game(handle)
            except (ValueError, UnicodeError) as exc:
                LOGGER.warning(
                    "Stopped parsing %s after game %d: %s",
                    pgn_path,
                    game_index,
                    exc,
                )
                return

            if game is None:
                return

            game_index += 1
            yield game_index, game


def parse_game(
    game: chess.pgn.Game,
    *,
    username: str,
    source_file: Path,
    source_game_index: int,
) -> ParsedGame | RejectedGame:
    """Validate and parse one game."""
    headers = game.headers
    source_name = source_file.name

    variant = headers.get("Variant", "").strip().lower()

    if variant not in STANDARD_VARIANTS:
        return RejectedGame(source_name, source_game_index, f"variant:{variant}")

    if game.errors:
        return RejectedGame(source_name, source_game_index, "pgn_parse_error")

    white = headers.get("White", "").strip()
    black = headers.get("Black", "").strip()

    normalized_username = normalize_username(username)
    normalized_white = normalize_username(white)
    normalized_black = normalize_username(black)

    if normalized_white == normalized_username:
        player_color = "white"
    elif normalized_black == normalized_username:
        player_color = "black"
    else:
        return RejectedGame(source_name, source_game_index, "player_not_found")

    try:
        board = game.board()
    except (ValueError, TypeError):
        return RejectedGame(
            source_name,
            source_game_index,
            "invalid_starting_position",
        )

    if type(board) is not chess.Board or board.chess960:
        return RejectedGame(
            source_name,
            source_game_index,
            "nonstandard_board",
        )

    standard_start = chess.Board()

    if board.fen() != standard_start.fen():
        return RejectedGame(
            source_name,
            source_game_index,
            "nonstandard_starting_position",
        )

    moves_uci: list[str] = []
    moves_san: list[str] = []

    try:
        for move in game.mainline_moves():
            if move not in board.legal_moves:
                return RejectedGame(source_name, source_game_index, "illegal_move")

            moves_san.append(board.san(move))
            moves_uci.append(move.uci())
            board.push(move)
    except (ValueError, AssertionError):
        return RejectedGame(source_name, source_game_index, "move_replay_error")

    if not moves_uci:
        return RejectedGame(source_name, source_game_index, "no_moves")

    result = headers.get("Result", "*").strip()

    white_rating = parse_optional_int(headers.get("WhiteElo") or headers.get("WhiteRating"))
    black_rating = parse_optional_int(headers.get("BlackElo") or headers.get("BlackRating"))

    if player_color == "white":
        player_rating = white_rating
        opponent_rating = black_rating
    else:
        player_rating = black_rating
        opponent_rating = white_rating

    game_id = make_game_id(
        source_file,
        source_game_index,
        headers,
        moves_uci,
    )

    return ParsedGame(
        game_id=game_id,
        source_file=source_name,
        source_game_index=source_game_index,
        event=headers.get("Event", ""),
        site=headers.get("Site", ""),
        date=headers.get("Date", ""),
        utc_date=headers.get("UTCDate", ""),
        utc_time=headers.get("UTCTime", ""),
        end_time=headers.get("EndTime", ""),
        time_control=headers.get("TimeControl", ""),
        time_class=infer_time_class(headers),
        rated=parse_optional_bool(headers.get("Rated")),
        result=result,
        termination=headers.get("Termination", ""),
        eco=headers.get("ECO", ""),
        opening=headers.get("Opening", ""),
        white_username=white,
        black_username=black,
        white_rating=white_rating,
        black_rating=black_rating,
        player_color=player_color,
        player_rating=player_rating,
        opponent_rating=opponent_rating,
        player_result=infer_player_result(result, player_color),
        ply_count=len(moves_uci),
        move_count=(len(moves_uci) + 1) // 2,
        moves_uci=moves_uci,
        moves_san=moves_san,
        final_fen=board.fen(),
    )


def build_corpus(
    input_dir: Path,
    username: str,
) -> tuple[list[ParsedGame], list[RejectedGame]]:
    """Parse all monthly PGNs in chronological filename order."""
    pgn_paths = sorted(input_dir.glob("*.pgn"))

    if not pgn_paths:
        raise FileNotFoundError(f"No PGN files found in {input_dir}")

    accepted: list[ParsedGame] = []
    rejected: list[RejectedGame] = []

    for pgn_path in pgn_paths:
        LOGGER.info("Parsing %s", pgn_path)

        for game_index, game in read_games(pgn_path):
            result = parse_game(
                game,
                username=username,
                source_file=pgn_path,
                source_game_index=game_index,
            )

            if isinstance(result, ParsedGame):
                accepted.append(result)
            else:
                rejected.append(result)

    return accepted, rejected


def write_corpus(games: list[ParsedGame], output_path: Path) -> None:
    """Write accepted games to Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [asdict(game) for game in games]
    dataframe = pd.DataFrame.from_records(records)

    dataframe.to_parquet(output_path, index=False)


def build_summary(
    games: list[ParsedGame],
    rejected: list[RejectedGame],
) -> dict[str, object]:
    """Build a human-readable audit summary."""
    rejection_counts = Counter(item.reason for item in rejected)
    time_class_counts = Counter(game.time_class for game in games)
    result_counts = Counter(game.player_result for game in games)
    color_counts = Counter(game.player_color for game in games)
    source_counts = Counter(game.source_file for game in games)

    ratings = [game.player_rating for game in games if game.player_rating is not None]

    return {
        "accepted_games": len(games),
        "rejected_games": len(rejected),
        "total_games_seen": len(games) + len(rejected),
        "total_player_positions": sum(
            game.move_count if game.player_color == "white" else game.ply_count // 2
            for game in games
        ),
        "total_plies": sum(game.ply_count for game in games),
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "time_classes": dict(sorted(time_class_counts.items())),
        "player_results": dict(sorted(result_counts.items())),
        "player_colors": dict(sorted(color_counts.items())),
        "accepted_games_by_archive": dict(sorted(source_counts.items())),
        "player_rating": {
            "minimum": min(ratings) if ratings else None,
            "maximum": max(ratings) if ratings else None,
            "mean": round(sum(ratings) / len(ratings), 2) if ratings else None,
            "available": len(ratings),
        },
    }


def write_summary(summary: dict[str, object], output_path: Path) -> None:
    """Write the corpus summary as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Parse and audit downloaded Chess.com PGN archives."
    )
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
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
    """Run the corpus parser CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = parse_args()

    games, rejected = build_corpus(
        input_dir=args.input_dir,
        username=args.username,
    )

    write_corpus(games, args.output)

    summary = build_summary(games, rejected)
    write_summary(summary, args.summary)

    print()
    print(f"Games accepted:       {summary['accepted_games']}")
    print(f"Games rejected:       {summary['rejected_games']}")
    print(f"Player move samples:  {summary['total_player_positions']}")
    print(f"Game dataset:         {args.output}")
    print(f"Corpus summary:       {args.summary}")

    rejection_reasons = summary["rejection_reasons"]

    if isinstance(rejection_reasons, dict) and rejection_reasons:
        print()
        print("Rejection reasons:")

        for reason, count in rejection_reasons.items():
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
