"""Tests for held-out hybrid-engine evaluation utilities."""

import chess
import pandas as pd
import pytest

from yeafins.engine.evaluate_hybrid import (
    build_summary,
    deterministic_sample,
    parse_style_weights,
    select_methods,
    unique_moves,
)
from yeafins.engine.hybrid import CandidateMove


def make_candidates() -> list[CandidateMove]:
    return [
        CandidateMove(
            move=chess.Move.from_uci("e2e4"),
            model_probability=0.60,
            model_rank=1,
            stockfish_cp=20,
        ),
        CandidateMove(
            move=chess.Move.from_uci("d2d4"),
            model_probability=0.25,
            model_rank=2,
            stockfish_cp=50,
        ),
        CandidateMove(
            move=chess.Move.from_uci("g1f3"),
            model_probability=0.15,
            model_rank=3,
            stockfish_cp=30,
        ),
    ]


def test_deterministic_sample_is_reproducible() -> None:
    dataframe = pd.DataFrame({"sample_id": [f"sample-{index}" for index in range(20)]})

    first = deterministic_sample(
        dataframe,
        sample_size=5,
        seed=42,
    )
    second = deterministic_sample(
        dataframe,
        sample_size=5,
        seed=42,
    )

    assert first["sample_id"].tolist() == second["sample_id"].tolist()


def test_unique_moves_preserves_order() -> None:
    e4 = chess.Move.from_uci("e2e4")
    d4 = chess.Move.from_uci("d2d4")

    assert unique_moves([e4, d4, e4]) == [e4, d4]


def test_select_methods() -> None:
    selected = select_methods(
        make_candidates(),
        top_k_values=(2, 3),
        style_weights=(0.0, 1.0),
    )

    assert selected["raw_policy"].move == chess.Move.from_uci("e2e4")
    assert selected["best_of_top_2"].move == chess.Move.from_uci("d2d4")
    assert selected["best_of_top_3"].move == chess.Move.from_uci("d2d4")
    assert selected["blended_0.00"].move == chess.Move.from_uci("d2d4")
    assert selected["blended_1.00"].move == chess.Move.from_uci("e2e4")


def test_parse_style_weights() -> None:
    assert parse_style_weights("0.25,0.5,0.75") == (
        0.25,
        0.5,
        0.75,
    )

    with pytest.raises(ValueError):
        parse_style_weights("")

    with pytest.raises(ValueError):
        parse_style_weights("1.5")


def test_build_summary() -> None:
    results = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "actual_cp": 10,
                "actual_in_top_3": True,
                "actual_in_top_5": True,
                "actual_in_top_8": True,
                "evaluation_seconds": 0.1,
                "raw_policy_cp": 10,
                "raw_policy_model_rank": 1,
                "raw_policy_model_probability": 0.6,
                "raw_policy_matches_actual": True,
                "best_of_top_3_cp": 40,
                "best_of_top_3_model_rank": 2,
                "best_of_top_3_model_probability": 0.3,
                "best_of_top_3_matches_actual": False,
            },
            {
                "game_id": "g2",
                "actual_cp": 0,
                "actual_in_top_3": False,
                "actual_in_top_5": True,
                "actual_in_top_8": True,
                "evaluation_seconds": 0.2,
                "raw_policy_cp": -20,
                "raw_policy_model_rank": 1,
                "raw_policy_model_probability": 0.5,
                "raw_policy_matches_actual": False,
                "best_of_top_3_cp": 10,
                "best_of_top_3_model_rank": 3,
                "best_of_top_3_model_probability": 0.2,
                "best_of_top_3_matches_actual": False,
            },
        ]
    )

    summary = build_summary(
        results,
        method_names=(
            "raw_policy",
            "best_of_top_3",
        ),
        metadata={"seed": 42},
    )

    assert summary["positions_evaluated"] == 2
    assert summary["candidate_coverage"]["top3"] == 0.5
    assert summary["methods"]["best_of_top_3"]["mean_cp_improvement_over_raw"] == 30.0
