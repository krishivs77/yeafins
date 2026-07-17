"""Run a one-position Yeafins hybrid-engine demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path

import chess

from yeafins.engine.hybrid import YeafinsHybridEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Choose a move using the Yeafins hybrid engine.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fen",
        default=chess.STARTING_FEN,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--mode",
        choices=["best_of_top_k", "blended"],
        default="best_of_top_k",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--style-weight",
        type=float,
        default=0.20,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    board = chess.Board(args.fen)

    with YeafinsHybridEngine(
        checkpoint_path=args.checkpoint,
    ) as engine:
        decision = engine.choose_move(
            board,
            top_k=args.top_k,
            mode=args.mode,
            depth=args.depth,
            style_weight=args.style_weight,
        )

    print(f"Selected move: {decision.selected_move.uci()}")
    print()
    print("Candidates:")

    for candidate in decision.candidates:
        blended = (
            "" if candidate.blended_score is None else f" blended={candidate.blended_score:.4f}"
        )

        print(
            f"  rank={candidate.model_rank} "
            f"move={candidate.move.uci()} "
            f"prob={candidate.model_probability:.4f} "
            f"stockfish={candidate.stockfish_cp:+d}cp"
            f"{blended}"
        )


if __name__ == "__main__":
    main()
