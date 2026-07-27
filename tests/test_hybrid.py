"""Tests for hybrid model and Stockfish move selection."""

from unittest.mock import MagicMock

import chess
import chess.engine
import pytest

from yeafins.engine.hybrid import (
    CandidateMove,
    HybridEngineError,
    choose_best_of_top_k,
    choose_blended,
    evaluate_candidates,
    evaluate_root_moves,
    infer_game_phase,
    normalize_stockfish_scores,
    phase_style_weight,
    validate_stockfish_elo,
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


def proposed_candidates(count: int) -> list[tuple[chess.Move, float, int]]:
    moves = list(chess.Board().legal_moves)[:count]
    return [(move, 1.0 / (index + 2), index + 1) for index, move in enumerate(moves)]


def multipv_results(
    candidates: list[tuple[chess.Move, float, int]],
) -> list[dict[str, object]]:
    return [
        {
            "pv": [move],
            "score": chess.engine.PovScore(
                chess.engine.Cp(index * 10),
                chess.WHITE,
            ),
        }
        for index, (move, _, _) in reversed(list(enumerate(candidates, start=1)))
    ]


def test_time_limited_candidates_use_one_restricted_multipv_call() -> None:
    board = chess.Board()
    candidates = proposed_candidates(16)
    engine = MagicMock()
    engine.analyse.return_value = multipv_results(candidates)

    evaluated = evaluate_candidates(
        engine,
        board,
        candidates,
        depth=None,
        time_limit_seconds=1.5,
    )

    engine.analyse.assert_called_once()
    called_board, limit = engine.analyse.call_args.args
    assert called_board is board
    assert limit.time == 1.5
    assert limit.depth is None
    assert engine.analyse.call_args.kwargs["root_moves"] == [move for move, _, _ in candidates]
    assert engine.analyse.call_args.kwargs["multipv"] == 16
    assert [candidate.move for candidate in evaluated] == [move for move, _, _ in candidates]
    assert [candidate.stockfish_cp for candidate in evaluated] == [
        index * 10 for index in range(1, 17)
    ]


def test_depth_limited_candidates_are_batched() -> None:
    board = chess.Board()
    candidates = proposed_candidates(4)
    engine = MagicMock()
    engine.analyse.return_value = multipv_results(candidates)

    evaluate_candidates(
        engine,
        board,
        candidates,
        depth=10,
        time_limit_seconds=None,
    )

    engine.analyse.assert_called_once()
    limit = engine.analyse.call_args.args[1]
    assert limit.depth == 10
    assert limit.time is None
    assert engine.analyse.call_args.kwargs["multipv"] == 4


@pytest.mark.parametrize("candidate_count", [1, 4, 16])
def test_analyse_call_count_does_not_scale_with_top_k(candidate_count: int) -> None:
    board = chess.Board()
    candidates = proposed_candidates(candidate_count)
    engine = MagicMock()
    engine.analyse.return_value = multipv_results(candidates)

    evaluate_candidates(
        engine,
        board,
        candidates,
        depth=None,
        time_limit_seconds=1.5,
    )

    assert engine.analyse.call_count == 1


def test_single_root_move_accepts_dict_result_and_preserves_mate_score() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    engine = MagicMock()
    engine.analyse.return_value = {
        "pv": [move],
        "score": chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE),
    }

    scores = evaluate_root_moves(
        engine,
        board,
        [move],
        depth=None,
        time_limit_seconds=1.5,
    )

    assert scores == {move: 99_997}


def test_missing_multipv_root_move_raises_engine_error() -> None:
    board = chess.Board()
    candidates = proposed_candidates(3)
    engine = MagicMock()
    engine.analyse.return_value = multipv_results(candidates[:2])

    with pytest.raises(HybridEngineError, match="did not score all root moves"):
        evaluate_candidates(
            engine,
            board,
            candidates,
            depth=None,
            time_limit_seconds=1.5,
        )


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


def test_validate_stockfish_elo() -> None:
    validate_stockfish_elo(1320)
    validate_stockfish_elo(2000)
    validate_stockfish_elo(3190)

    with pytest.raises(ValueError):
        validate_stockfish_elo(1319)

    with pytest.raises(ValueError):
        validate_stockfish_elo(3191)
