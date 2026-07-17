"""Play complete chess games against the Yeafins hybrid engine."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import chess
import chess.pgn

from yeafins.engine.hybrid import (
    HybridDecision,
    SelectionMode,
    YeafinsHybridEngine,
)

PlayerColor = Literal["white", "black"]


@dataclass(frozen=True)
class GameConfig:
    """Configuration for an interactive Yeafins game."""

    checkpoint_path: Path
    player_color: PlayerColor = "white"
    top_k: int = 8
    mode: SelectionMode = "blended"
    depth: int | None = 10
    time_limit_seconds: float | None = None
    style_weight: float | None = None
    show_candidates: bool = True
    output_directory: Path = Path("games")


def parse_move(
    board: chess.Board,
    move_text: str,
) -> chess.Move:
    """Parse a legal move written in SAN or UCI notation."""
    cleaned = move_text.strip()

    if not cleaned:
        raise ValueError("Move cannot be empty")

    try:
        move = board.parse_san(cleaned)
    except ValueError:
        try:
            move = chess.Move.from_uci(cleaned.lower())
        except ValueError as error:
            raise ValueError(f"Could not understand move: {cleaned}") from error

        if move not in board.legal_moves:
            raise ValueError(f"Move is illegal in this position: {cleaned}")

    return move


def format_board(board: chess.Board) -> str:
    """Render the board with coordinates."""
    rows = str(board).splitlines()

    labelled_rows = [f"{8 - index}  {row}" for index, row in enumerate(rows)]

    labelled_rows.append("")
    labelled_rows.append("   a b c d e f g h")

    return "\n".join(labelled_rows)


def print_candidates(
    board: chess.Board,
    decision: HybridDecision,
) -> None:
    """Print the hybrid engine's considered candidates."""
    print()
    print("Yeafins candidates")

    for candidate in decision.candidates:
        san = board.san(candidate.move)

        print(
            f"  rank={candidate.model_rank:<2d} "
            f"move={candidate.move.uci():<5s} "
            f"san={san:<7s} "
            f"prob={candidate.model_probability:.3f} "
            f"stockfish={candidate.stockfish_cp:+d}cp"
        )


def result_description(board: chess.Board) -> str:
    """Return a readable description of the game outcome."""
    outcome = board.outcome(claim_draw=True)

    if outcome is None:
        return "Game unfinished"

    termination_names = {
        chess.Termination.CHECKMATE: "checkmate",
        chess.Termination.STALEMATE: "stalemate",
        chess.Termination.INSUFFICIENT_MATERIAL: ("insufficient material"),
        chess.Termination.SEVENTYFIVE_MOVES: ("seventy-five-move rule"),
        chess.Termination.FIVEFOLD_REPETITION: ("fivefold repetition"),
        chess.Termination.FIFTY_MOVES: "fifty-move rule",
        chess.Termination.THREEFOLD_REPETITION: ("threefold repetition"),
        chess.Termination.VARIANT_WIN: "variant win",
        chess.Termination.VARIANT_LOSS: "variant loss",
        chess.Termination.VARIANT_DRAW: "variant draw",
    }

    reason = termination_names.get(
        outcome.termination,
        outcome.termination.name.lower().replace("_", " "),
    )

    if outcome.winner is chess.WHITE:
        winner = "White"
    elif outcome.winner is chess.BLACK:
        winner = "Black"
    else:
        winner = "Draw"

    return f"{winner} by {reason}"


def create_pgn(
    board: chess.Board,
    *,
    player_color: PlayerColor,
) -> chess.pgn.Game:
    """Create a PGN record from the completed board."""
    game = chess.pgn.Game.from_board(board)

    timestamp = datetime.now().astimezone()

    game.headers["Event"] = "Human vs Yeafins"
    game.headers["Site"] = "Local terminal"
    game.headers["Date"] = timestamp.strftime("%Y.%m.%d")
    game.headers["Round"] = "1"

    if player_color == "white":
        game.headers["White"] = "Krishiv"
        game.headers["Black"] = "Yeafins"
    else:
        game.headers["White"] = "Yeafins"
        game.headers["Black"] = "Krishiv"

    outcome = board.outcome(claim_draw=True)

    game.headers["Result"] = outcome.result() if outcome is not None else "*"

    return game


def save_pgn(
    board: chess.Board,
    *,
    player_color: PlayerColor,
    output_directory: Path,
) -> Path:
    """Save the game to a timestamped PGN file."""
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

    output_path = output_directory / (f"yeafins_game_{timestamp}.pgn")

    game = create_pgn(
        board,
        player_color=player_color,
    )

    output_path.write_text(
        str(game) + "\n",
        encoding="utf-8",
    )

    return output_path


def human_turn(board: chess.Board) -> bool:
    """Read and apply one human move.

    Returns False when the player requests to quit.
    """
    while True:
        move_text = input("\nYour move (SAN/UCI, or 'quit'): ").strip()

        if move_text.lower() in {
            "quit",
            "exit",
            "resign",
        }:
            return False

        try:
            move = parse_move(
                board,
                move_text,
            )
        except ValueError as error:
            print(error)
            continue

        print(f"You played: {board.san(move)} ({move.uci()})")

        board.push(move)
        return True


def engine_turn(
    board: chess.Board,
    engine: YeafinsHybridEngine,
    config: GameConfig,
) -> None:
    """Choose and apply one Yeafins move."""
    decision = engine.choose_move(
        board,
        top_k=config.top_k,
        mode=config.mode,
        depth=config.depth,
        time_limit_seconds=config.time_limit_seconds,
        style_weight=config.style_weight,
    )

    if config.show_candidates:
        print_candidates(
            board,
            decision,
        )

    selected_move = decision.selected_move
    san = board.san(selected_move)

    print()
    print(f"Yeafins plays: {san} ({selected_move.uci()})")

    board.push(selected_move)


def play_game(config: GameConfig) -> Path:
    """Play one complete interactive game."""
    if config.player_color not in {
        "white",
        "black",
    }:
        raise ValueError("player_color must be 'white' or 'black'")

    board = chess.Board()

    human_is_white = config.player_color == "white"

    print("Yeafins")
    print("Personalized chess policy + Stockfish")
    print()
    print(f"You are playing as {config.player_color.capitalize()}.")
    print("Enter moves using SAN (e4, Nf3, O-O) or UCI (e2e4).")
    print()

    with YeafinsHybridEngine(
        checkpoint_path=config.checkpoint_path,
    ) as engine:
        while not board.is_game_over(claim_draw=True):
            print()
            print(format_board(board))

            human_to_move = (board.turn == chess.WHITE and human_is_white) or (
                board.turn == chess.BLACK and not human_is_white
            )

            if human_to_move:
                continued = human_turn(board)

                if not continued:
                    print()
                    print("You ended the game.")
                    break
            else:
                engine_turn(
                    board,
                    engine,
                    config,
                )

    print()
    print(format_board(board))
    print()
    print(result_description(board))

    output_path = save_pgn(
        board,
        player_color=config.player_color,
        output_directory=config.output_directory,
    )

    print(f"PGN saved to: {output_path}")

    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Play a complete chess game against the Yeafins hybrid engine.")
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--color",
        choices=["white", "black"],
        default="white",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--mode",
        choices=[
            "best_of_top_k",
            "blended",
        ],
        default="blended",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--time-limit-seconds",
        type=float,
    )
    parser.add_argument(
        "--style-weight",
        type=float,
        default=None,
        help=("Override phase-aware style weighting."),
    )
    parser.add_argument(
        "--hide-candidates",
        action="store_true",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("games"),
    )

    return parser.parse_args()


def main() -> None:
    """Run a terminal game."""
    args = parse_args()

    depth = None if args.time_limit_seconds is not None else args.depth

    config = GameConfig(
        checkpoint_path=args.checkpoint,
        player_color=args.color,
        top_k=args.top_k,
        mode=args.mode,
        depth=depth,
        time_limit_seconds=args.time_limit_seconds,
        style_weight=args.style_weight,
        show_candidates=not args.hide_candidates,
        output_directory=args.output_directory,
    )

    play_game(config)


if __name__ == "__main__":
    main()
