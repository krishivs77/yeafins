"""Compare fixed and phase-aware hybrid weighting using saved evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from yeafins.engine.hybrid import CandidateMove, choose_blended

DEFAULT_INPUT_PATH = Path("runs/resnet_baseline/hybrid_full/hybrid_positions.parquet")
DEFAULT_OUTPUT_PATH = Path("runs/resnet_baseline/phase_weight_comparison.json")

PHASE_WEIGHTS = {
    "opening": 0.30,
    "middlegame": 0.10,
    "endgame": 0.20,
}

CP_CLIP = 1_000


def reconstruct_candidates(row: pd.Series) -> list[CandidateMove]:
    """Reconstruct the model's top-eight candidates from a saved row."""
    candidates: list[CandidateMove] = []

    for rank in range(1, 9):
        move_column = f"candidate_{rank}_move"
        probability_column = f"candidate_{rank}_probability"
        cp_column = f"candidate_{rank}_cp"

        if move_column not in row.index:
            raise KeyError(
                f"Saved evaluation does not contain per-candidate columns. Missing {move_column}."
            )

        import chess

        candidates.append(
            CandidateMove(
                move=chess.Move.from_uci(str(row[move_column])),
                model_probability=float(row[probability_column]),
                model_rank=rank,
                stockfish_cp=int(row[cp_column]),
            )
        )

    return candidates


def select_saved_candidate(
    candidates: list[CandidateMove],
    style_weight: float,
) -> CandidateMove:
    """Select a candidate using the current normalized blend."""
    return choose_blended(
        candidates,
        style_weight=style_weight,
    )


def summarize(
    dataframe: pd.DataFrame,
    *,
    move_column: str,
    cp_column: str,
    rank_column: str,
) -> dict[str, float | int]:
    """Summarize one reconstructed strategy."""
    selected_cp = dataframe[cp_column].astype(float)
    raw_cp = dataframe["raw_policy_cp"].astype(float)
    best_cp = dataframe["best_candidate_cp"].astype(float)

    clipped_selected = selected_cp.clip(-CP_CLIP, CP_CLIP)
    clipped_raw = raw_cp.clip(-CP_CLIP, CP_CLIP)
    clipped_best = best_cp.clip(-CP_CLIP, CP_CLIP)

    loss_to_best = clipped_best - clipped_selected

    return {
        "samples": int(len(dataframe)),
        "actual_move_match_rate": float(
            (dataframe[move_column] == dataframe["actual_move"]).mean()
        ),
        "mean_selected_model_rank": float(dataframe[rank_column].mean()),
        "mean_clipped_cp_gain_over_raw": float((clipped_selected - clipped_raw).mean()),
        "mean_clipped_cp_loss_to_best": float(loss_to_best.mean()),
        "loss_to_best_at_least_100cp_rate": float((loss_to_best >= 100).mean()),
        "loss_to_best_at_least_200cp_rate": float((loss_to_best >= 200).mean()),
    }


def compare_weights(
    input_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Compare fixed 0.20 and phase-aware style weights."""
    dataframe = pd.read_parquet(input_path)

    required_candidate_columns = {
        f"candidate_{rank}_{suffix}"
        for rank in range(1, 9)
        for suffix in ("move", "probability", "cp")
    }

    missing = required_candidate_columns - set(dataframe.columns)

    if missing:
        rendered = ", ".join(sorted(missing)[:5])

        raise KeyError(
            "The existing evaluation file does not store individual "
            f"candidate records. Missing examples: {rendered}"
        )

    fixed_moves: list[str] = []
    fixed_cps: list[int] = []
    fixed_ranks: list[int] = []

    phase_moves: list[str] = []
    phase_cps: list[int] = []
    phase_ranks: list[int] = []
    resolved_weights: list[float] = []

    for _, row in dataframe.iterrows():
        candidates = reconstruct_candidates(row)

        fixed = select_saved_candidate(
            candidates,
            style_weight=0.20,
        )

        phase = str(row["game_phase"])
        phase_weight = PHASE_WEIGHTS[phase]

        phase_aware = select_saved_candidate(
            candidates,
            style_weight=phase_weight,
        )

        fixed_moves.append(fixed.move.uci())
        fixed_cps.append(fixed.stockfish_cp)
        fixed_ranks.append(fixed.model_rank)

        phase_moves.append(phase_aware.move.uci())
        phase_cps.append(phase_aware.stockfish_cp)
        phase_ranks.append(phase_aware.model_rank)
        resolved_weights.append(phase_weight)

    dataframe["fixed_020_move"] = fixed_moves
    dataframe["fixed_020_cp"] = fixed_cps
    dataframe["fixed_020_rank"] = fixed_ranks

    dataframe["phase_aware_move"] = phase_moves
    dataframe["phase_aware_cp"] = phase_cps
    dataframe["phase_aware_rank"] = phase_ranks
    dataframe["phase_style_weight"] = resolved_weights

    results: dict[str, object] = {
        "input_path": str(input_path),
        "phase_weights": PHASE_WEIGHTS,
        "overall": {
            "fixed_0.20": summarize(
                dataframe,
                move_column="fixed_020_move",
                cp_column="fixed_020_cp",
                rank_column="fixed_020_rank",
            ),
            "phase_aware": summarize(
                dataframe,
                move_column="phase_aware_move",
                cp_column="phase_aware_cp",
                rank_column="phase_aware_rank",
            ),
        },
        "by_phase": {},
    }

    for phase, group in dataframe.groupby(
        "game_phase",
        observed=True,
    ):
        results["by_phase"][str(phase)] = {
            "fixed_0.20": summarize(
                group,
                move_column="fixed_020_move",
                cp_column="fixed_020_cp",
                rank_column="fixed_020_rank",
            ),
            "phase_aware": summarize(
                group,
                move_column="phase_aware_move",
                cp_column="phase_aware_cp",
                rank_column="phase_aware_rank",
            ),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare fixed and phase-aware hybrid weights.")
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
    return parser.parse_args()


def main() -> None:
    """Run the saved-candidate comparison."""
    args = parse_args()

    results = compare_weights(
        input_path=args.input,
        output_path=args.output,
    )

    for strategy, metrics in results["overall"].items():
        print(
            f"{strategy:12s} "
            f"match={metrics['actual_move_match_rate']:.3f} "
            f"rank={metrics['mean_selected_model_rank']:.2f} "
            f"gain={metrics['mean_clipped_cp_gain_over_raw']:+.1f}cp "
            f"loss_best={metrics['mean_clipped_cp_loss_to_best']:.1f}cp"
        )


if __name__ == "__main__":
    main()
