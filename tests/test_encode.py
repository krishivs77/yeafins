"""Tests for AlphaZero-style chess move encoding."""

import chess
import numpy as np
import pytest

from yeafins.data.encode import (
    POLICY_SIZE,
    MoveEncodingError,
    decode_move,
    encode_move,
    legal_move_mask,
)


def assert_round_trip(board: chess.Board, move: chess.Move) -> None:
    """Assert that a legal move survives encode/decode conversion."""
    encoded = encode_move(move, board)
    decoded = decode_move(encoded, board)

    assert decoded == move
    assert 0 <= encoded < POLICY_SIZE


def test_policy_size() -> None:
    assert POLICY_SIZE == 4672


def test_round_trip_all_starting_moves() -> None:
    board = chess.Board()

    for move in list(board.legal_moves):
        assert_round_trip(board, move)


def test_round_trip_castling() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")

    assert_round_trip(board, chess.Move.from_uci("e1g1"))
    assert_round_trip(board, chess.Move.from_uci("e1c1"))


def test_round_trip_white_promotions() -> None:
    board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")

    for promotion in ("q", "r", "b", "n"):
        assert_round_trip(
            board,
            chess.Move.from_uci(f"a7a8{promotion}"),
        )


def test_round_trip_black_promotions() -> None:
    board = chess.Board("7k/8/8/8/8/8/p7/7K b - - 0 1")

    for promotion in ("q", "r", "b", "n"):
        assert_round_trip(
            board,
            chess.Move.from_uci(f"a2a1{promotion}"),
        )


def test_round_trip_capture_underpromotions() -> None:
    white_board = chess.Board("1r5k/P7/8/8/8/8/8/7K w - - 0 1")

    for promotion in ("r", "b", "n"):
        assert_round_trip(
            white_board,
            chess.Move.from_uci(f"a7b8{promotion}"),
        )

    black_board = chess.Board("7k/8/8/8/8/8/p7/1R5K b - - 0 1")

    for promotion in ("r", "b", "n"):
        assert_round_trip(
            black_board,
            chess.Move.from_uci(f"a2b1{promotion}"),
        )


def test_legal_move_mask_matches_legal_moves() -> None:
    board = chess.Board()

    mask = legal_move_mask(board)

    assert mask.dtype == np.bool_
    assert mask.shape == (POLICY_SIZE,)
    assert int(mask.sum()) == board.legal_moves.count()

    for move in board.legal_moves:
        assert mask[encode_move(move, board)]


def test_encode_rejects_illegal_move() -> None:
    board = chess.Board()
    illegal_move = chess.Move.from_uci("e2e5")

    with pytest.raises(MoveEncodingError, match="illegal"):
        encode_move(illegal_move, board)


def test_decode_rejects_out_of_range_index() -> None:
    board = chess.Board()

    with pytest.raises(MoveEncodingError):
        decode_move(-1, board)

    with pytest.raises(MoveEncodingError):
        decode_move(POLICY_SIZE, board)


def test_all_legal_moves_have_unique_indices() -> None:
    board = chess.Board("r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2N1BN2/PPP2PPP/R2QK2R w KQkq - 4 10")

    indices = [encode_move(move, board) for move in board.legal_moves]

    assert len(indices) == len(set(indices))
