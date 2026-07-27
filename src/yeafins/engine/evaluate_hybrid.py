"""Evaluate personalized Stockfish-hybrid move selection on held-out positions."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import chess
import chess.engine
import pandas as pd
import torch
from tqdm import tqdm

from yeafins.engine.hybrid import (
    CandidateMove,
    choose_best_of_top_k,
    choose_blended,
    evaluate_root_moves,
    load_policy_model,
    model_candidates,
    resolve_stockfish_path,
)

DEFAULT_POSITIONS_PATH = Path("data/processed/positions.parquet")
DEFAULT_CHECKPOINT_PATH = Path("runs/resnet_baseline/best.pt")
DEFAULT_OUTPUT_DIR = Path("runs/resnet_baseline/hybrid_evaluation")

DEFAULT_SAMPLE_SIZE = 500
DEFAULT_SEED = 42
DEFAULT_MAX_K = 8
DEFAULT_DEPTH = 10
DEFAULT_STYLE_WEIGHTS = (0.25, 0.50, 0.75)

CP_CLIP = 1_000
BLUNDER_THRESHOLDS = (100, 200, 500)


class HybridEvaluationError(RuntimeError):
    """Raised when hybrid evaluation cannot be completed safely."""


def deterministic_sample(
    positions: pd.DataFrame,
    *,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    """Take a reproducible random position-level sample."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    if positions.empty:
        raise ValueError("Cannot sample from an empty dataset")

    effective_size = min(sample_size, len(positions))

    return (
        positions.sample(
            n=effective_size,
            replace=False,
            random_state=seed,
        )
        .sort_values("sample_id")
        .reset_index(drop=True)
    )


def unique_moves(moves: Iterable[chess.Move]) -> list[chess.Move]:
    """Deduplicate moves while preserving their original order."""
    seen: set[chess.Move] = set()
    result: list[chess.Move] = []

    for move in moves:
        if move not in seen:
            seen.add(move)
            result.append(move)

    return result


def game_phase(fen: str) -> str:
    """Assign a simple opening, middlegame, or endgame label."""
    board = chess.Board(fen)

    if board.fullmove_number <= 10:
        return "opening"

    values = {
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    non_pawn_material = sum(
        len(board.pieces(piece_type, color)) * value
        for piece_type, value in values.items()
        for color in (chess.WHITE, chess.BLACK)
    )

    if non_pawn_material <= 14:
        return "endgame"

    return "middlegame"


def candidate_records(
    proposed: list[tuple[chess.Move, float, int]],
    scores: dict[chess.Move, int],
) -> list[CandidateMove]:
    """Combine model candidates with their Stockfish evaluations."""
    return [
        CandidateMove(
            move=move,
            model_probability=probability,
            model_rank=rank,
            stockfish_cp=scores[move],
        )
        for move, probability, rank in proposed
    ]


def select_methods(
    candidates: list[CandidateMove],
    *,
    top_k_values: tuple[int, ...],
    style_weights: tuple[float, ...],
) -> dict[str, CandidateMove]:
    """Run all requested hybrid selection strategies."""
    if not candidates:
        raise ValueError("At least one candidate is required")

    selected: dict[str, CandidateMove] = {
        "raw_policy": candidates[0],
    }

    for top_k in top_k_values:
        effective_k = min(top_k, len(candidates))

        selected[f"best_of_top_{top_k}"] = choose_best_of_top_k(candidates[:effective_k])

    for style_weight in style_weights:
        label = f"blended_{style_weight:.2f}"

        selected[label] = choose_blended(
            candidates,
            style_weight=style_weight,
        )

    return selected


def evaluate_position(
    row: Any,
    *,
    model: torch.nn.Module,
    device: torch.device,
    engine: chess.engine.SimpleEngine,
    max_k: int,
    top_k_values: tuple[int, ...],
    style_weights: tuple[float, ...],
    temperature: float,
    depth: int | None,
    time_limit_seconds: float | None,
) -> dict[str, object]:
    """Evaluate all hybrid methods on one position."""
    fen = str(row.fen)
    board = chess.Board(fen)
    actual_move = chess.Move.from_uci(str(row.move_uci))

    if actual_move not in board.legal_moves:
        raise HybridEvaluationError(f"Stored actual move is illegal: {actual_move.uci()}")

    started = time.perf_counter()

    proposed = model_candidates(
        model,
        board,
        device=device,
        top_k=max_k,
        temperature=temperature,
    )

    root_moves = unique_moves(
        [
            *(move for move, _, _ in proposed),
            actual_move,
        ]
    )

    scores = evaluate_root_moves(
        engine,
        board,
        root_moves,
        depth=depth,
        time_limit_seconds=time_limit_seconds,
    )

    candidates = candidate_records(proposed, scores)

    best_candidate_cp = max(candidate.stockfish_cp for candidate in candidates)

    selections = select_methods(
        candidates,
        top_k_values=top_k_values,
        style_weights=style_weights,
    )

    actual_rank = next(
        (rank for move, _, rank in proposed if move == actual_move),
        None,
    )

    record: dict[str, object] = {
        "sample_id": str(row.sample_id),
        "game_id": str(row.game_id),
        "fen": fen,
        "actual_move": actual_move.uci(),
        "actual_cp": scores[actual_move],
        "best_candidate_cp": best_candidate_cp,
        "actual_model_rank": actual_rank,
        "actual_in_top_3": (actual_rank is not None and actual_rank <= 3),
        "actual_in_top_5": (actual_rank is not None and actual_rank <= 5),
        "actual_in_top_8": (actual_rank is not None and actual_rank <= 8),
        "time_class": str(row.time_class),
        "player_color": str(row.player_color),
        "player_rating": float(row.player_rating),
        "game_phase": game_phase(fen),
        "evaluation_seconds": (time.perf_counter() - started),
    }

    for method_name, candidate in selections.items():
        record[f"{method_name}_move"] = candidate.move.uci()
        record[f"{method_name}_cp"] = candidate.stockfish_cp
        record[f"{method_name}_model_rank"] = candidate.model_rank
        record[f"{method_name}_model_probability"] = candidate.model_probability
        record[f"{method_name}_matches_actual"] = candidate.move == actual_move

    return record


def clip_centipawns(series: pd.Series, limit: int = CP_CLIP) -> pd.Series:
    """Clip extreme mate-equivalent scores for robust aggregation."""
    return series.clip(lower=-limit, upper=limit)


def candidate_best_cp(results: pd.DataFrame) -> pd.Series:
    """Return the strongest model-proposed candidate score per position."""
    return results["best_candidate_cp"].astype(float)


def summarize_method(
    results: pd.DataFrame,
    method_name: str,
) -> dict[str, float | int]:
    """Summarize style fidelity and robust engine-strength metrics."""
    cp_column = f"{method_name}_cp"
    rank_column = f"{method_name}_model_rank"
    probability_column = f"{method_name}_model_probability"
    match_column = f"{method_name}_matches_actual"

    raw_cp = results["raw_policy_cp"].astype(float)
    method_cp = results[cp_column].astype(float)
    actual_cp = results["actual_cp"].astype(float)
    best_cp = candidate_best_cp(results).astype(float)

    improvement_over_raw = method_cp - raw_cp
    improvement_over_actual = method_cp - actual_cp
    loss_to_best = best_cp - method_cp

    clipped_method = clip_centipawns(method_cp)
    clipped_raw = clip_centipawns(raw_cp)
    clipped_actual = clip_centipawns(actual_cp)
    clipped_best = clip_centipawns(best_cp)

    clipped_improvement_over_raw = clipped_method - clipped_raw
    clipped_improvement_over_actual = clipped_method - clipped_actual
    clipped_loss_to_best = clipped_best - clipped_method

    summary: dict[str, float | int] = {
        "samples": int(len(results)),
        "actual_move_match_rate": float(results[match_column].mean()),
        "mean_model_rank": float(results[rank_column].mean()),
        "mean_model_probability": float(results[probability_column].mean()),
        # Robust strength summaries.
        "median_cp_improvement_over_raw": float(improvement_over_raw.median()),
        "mean_clipped_cp_improvement_over_raw": float(clipped_improvement_over_raw.mean()),
        "median_cp_improvement_over_actual": float(improvement_over_actual.median()),
        "mean_clipped_cp_improvement_over_actual": float(clipped_improvement_over_actual.mean()),
        "mean_clipped_cp_loss_to_best_candidate": float(clipped_loss_to_best.mean()),
        "median_cp_loss_to_best_candidate": float(loss_to_best.median()),
        "improved_over_raw_rate": float((improvement_over_raw > 0).mean()),
        "equal_to_raw_rate": float((improvement_over_raw == 0).mean()),
        "worsened_vs_raw_rate": float((improvement_over_raw < 0).mean()),
        # Retain uncapped values for debugging only.
        "diagnostic_mean_uncapped_cp": float(method_cp.mean()),
        "diagnostic_mean_uncapped_improvement_over_raw": float(improvement_over_raw.mean()),
    }

    for threshold in BLUNDER_THRESHOLDS:
        summary[f"loss_to_best_at_least_{threshold}cp_rate"] = float(
            (loss_to_best >= threshold).mean()
        )

    return summary


def build_summary(
    results: pd.DataFrame,
    *,
    method_names: tuple[str, ...],
    metadata: dict[str, object],
) -> dict[str, object]:
    """Build the complete hybrid-evaluation summary."""
    return {
        **metadata,
        "positions_evaluated": int(len(results)),
        "unique_games": int(results["game_id"].nunique()),
        "candidate_coverage": {
            "top3": float(results["actual_in_top_3"].mean()),
            "top5": float(results["actual_in_top_5"].mean()),
            "top8": float(results["actual_in_top_8"].mean()),
        },
        "runtime": {
            "total_seconds": float(results["evaluation_seconds"].sum()),
            "mean_seconds_per_position": float(results["evaluation_seconds"].mean()),
            "median_seconds_per_position": float(results["evaluation_seconds"].median()),
        },
        "methods": {
            method_name: summarize_method(
                results,
                method_name,
            )
            for method_name in method_names
        },
    }


def write_outputs(
    results: pd.DataFrame,
    summary: dict[str, object],
    output_dir: Path,
) -> None:
    """Persist detailed position records and aggregate metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_parquet(
        output_dir / "hybrid_positions.parquet",
        index=False,
    )

    results.to_csv(
        output_dir / "hybrid_positions.csv",
        index=False,
    )

    (output_dir / "hybrid_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_style_weights(value: str) -> tuple[float, ...]:
    """Parse comma-separated style weights."""
    weights = tuple(float(item.strip()) for item in value.split(",") if item.strip())

    if not weights:
        raise ValueError("At least one style weight is required")

    if any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError("Style weights must be between 0 and 1")

    return weights


def evaluate_hybrid(
    *,
    checkpoint_path: Path,
    positions_path: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
    max_k: int,
    depth: int | None,
    time_limit_seconds: float | None,
    temperature: float,
    style_weights: tuple[float, ...],
    threads: int,
    hash_mb: int,
    stockfish_path: str | None,
) -> dict[str, object]:
    """Run the held-out hybrid-engine experiment."""
    if max_k < 8:
        raise ValueError("max_k must be at least 8 for top-8 evaluation")

    columns = [
        "sample_id",
        "game_id",
        "split",
        "fen",
        "move_uci",
        "time_class",
        "player_color",
        "player_rating",
    ]

    positions = pd.read_parquet(
        positions_path,
        columns=columns,
    )

    positions = positions.loc[positions["split"] == "test"].reset_index(drop=True)

    sampled = deterministic_sample(
        positions,
        sample_size=sample_size,
        seed=seed,
    )

    model, device = load_policy_model(checkpoint_path)

    executable = resolve_stockfish_path(stockfish_path)

    engine = chess.engine.SimpleEngine.popen_uci(executable)
    engine.configure(
        {
            "Threads": threads,
            "Hash": hash_mb,
        }
    )

    records: list[dict[str, object]] = []

    try:
        for row in tqdm(
            sampled.itertuples(index=False),
            total=len(sampled),
            desc="Evaluating hybrid positions",
        ):
            records.append(
                evaluate_position(
                    row,
                    model=model,
                    device=device,
                    engine=engine,
                    max_k=max_k,
                    top_k_values=(3, 5, 8),
                    style_weights=style_weights,
                    temperature=temperature,
                    depth=depth,
                    time_limit_seconds=time_limit_seconds,
                )
            )
    finally:
        engine.quit()

    results = pd.DataFrame.from_records(records)

    method_names = (
        "raw_policy",
        "best_of_top_3",
        "best_of_top_5",
        "best_of_top_8",
        *tuple(f"blended_{weight:.2f}" for weight in style_weights),
    )

    summary = build_summary(
        results,
        method_names=method_names,
        metadata={
            "checkpoint": str(checkpoint_path),
            "positions_path": str(positions_path),
            "device": str(device),
            "stockfish_path": executable,
            "sample_size_requested": sample_size,
            "sample_seed": seed,
            "max_k": max_k,
            "depth": depth,
            "time_limit_seconds": time_limit_seconds,
            "temperature": temperature,
            "style_weights": list(style_weights),
            "threads": threads,
            "hash_mb": hash_mb,
        },
    )

    write_outputs(results, summary, output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=("Evaluate Stockfish selection among personalized policy candidates.")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--positions",
        type=Path,
        default=DEFAULT_POSITIONS_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=DEFAULT_MAX_K,
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
    )
    parser.add_argument(
        "--time-limit-seconds",
        type=float,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--style-weights",
        default=",".join(str(weight) for weight in DEFAULT_STYLE_WEIGHTS),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--hash-mb",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--stockfish-path",
    )
    return parser.parse_args()


def main() -> None:
    """Run hybrid evaluation from the command line."""
    args = parse_args()

    style_weights = parse_style_weights(args.style_weights)

    depth = None if args.time_limit_seconds is not None else args.depth

    summary = evaluate_hybrid(
        checkpoint_path=args.checkpoint,
        positions_path=args.positions,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        max_k=args.max_k,
        depth=depth,
        time_limit_seconds=args.time_limit_seconds,
        temperature=args.temperature,
        style_weights=style_weights,
        threads=args.threads,
        hash_mb=args.hash_mb,
        stockfish_path=args.stockfish_path,
    )

    print()
    print(
        "Positions evaluated:",
        summary["positions_evaluated"],
    )
    print(
        "Mean seconds/position:",
        f"{summary['runtime']['mean_seconds_per_position']:.3f}",
    )

    coverage = summary["candidate_coverage"]

    print()
    print("Candidate coverage")
    print(f"  Top 3: {coverage['top3']:.3f}")
    print(f"  Top 5: {coverage['top5']:.3f}")
    print(f"  Top 8: {coverage['top8']:.3f}")

    print()
    print("Selection methods")

    for method_name, metrics in summary["methods"].items():
        print(
            f"  {method_name:16s} "
            f"match={metrics['actual_move_match_rate']:.3f} "
            f"mean_rank={metrics['mean_model_rank']:.2f} "
            f"clipped_gain="
            f"{metrics['mean_clipped_cp_improvement_over_raw']:+.1f}cp "
            f"loss_to_best="
            f"{metrics['mean_clipped_cp_loss_to_best_candidate']:.1f}cp"
        )


if __name__ == "__main__":
    main()
