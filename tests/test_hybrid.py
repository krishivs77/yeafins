"""Tests for hybrid model and Stockfish move selection."""

import chess
import pytest

from yeafins.engine.hybrid import (
    CandidateMove,
    HybridEngineError,
    choose_best_of_top_k,
    choose_blended,
    infer_game_phase,
    normalize_stockfish_scores,
    phase_style_weight,
)


def make_candidates() -> list[CandidateMove]:

    return [
        CandidateMove(
            move=chess.Move.from_uci("e2e4"),
            model_probability=0.50,
            model_rank=1,
            stockfish_cp=30,
        ),
        CandidateMove(
            move=chess.Move.from_uci("d2d4"),
            model_probability=0.30,
            model_rank=2,
            stockfish_cp=45,
        ),
        CandidateMove(
            move=chess.Move.from_uci("g1f3"),
            model_probability=0.20,
            model_rank=3,
            stockfish_cp=20,
        ),
    ]


def test_normalize_stockfish_scores() -> None:
    normalized = normalize_stockfish_scores(make_candidates())

    assert normalized[chess.Move.from_uci("g1f3")] == 0.0
    assert normalized[chess.Move.from_uci("d2d4")] == 1.0


def test_choose_best_of_top_k() -> None:
    selected = choose_best_of_top_k(make_candidates())

    assert selected.move == chess.Move.from_uci("d2d4")


def test_choose_blended_favors_model_when_style_weight_is_high() -> None:
    selected = choose_blended(
        make_candidates(),
        style_weight=0.95,
    )

    assert selected.move == chess.Move.from_uci("e2e4")


def test_choose_blended_favors_engine_when_style_weight_is_low() -> None:
    selected = choose_blended(
        make_candidates(),
        style_weight=0.0,
    )

    assert selected.move == chess.Move.from_uci("d2d4")


def test_choose_functions_reject_empty_candidates() -> None:
    with pytest.raises(HybridEngineError):
        choose_best_of_top_k([])

    with pytest.raises(HybridEngineError):
        choose_blended([])


def test_infer_game_phase_opening() -> None:
    board = chess.Board()

    assert infer_game_phase(board) == "opening"
    assert phase_style_weight(board) == 0.20


def test_infer_game_phase_middlegame() -> None:
    board = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 15")

    assert infer_game_phase(board) == "middlegame"
    assert phase_style_weight(board) == 0.10


def test_infer_game_phase_endgame() -> None:
    board = chess.Board("8/8/4k3/8/8/4K3/4P3/8 w - - 0 30")

    assert infer_game_phase(board) == "endgame"
    assert phase_style_weight(board) == 0.20
