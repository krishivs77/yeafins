"""Tests for chess-board tensor encoding."""

import chess
import numpy as np
import pytest

from yeafins.data.board import (
    BLACK_KINGSIDE_CASTLING_PLANE,
    BLACK_QUEENSIDE_CASTLING_PLANE,
    BOARD_SHAPE,
    EN_PASSANT_PLANE,
    HALFMOVE_CLOCK_PLANE,
    PIECE_PLANES,
    SIDE_TO_MOVE_PLANE,
    WHITE_KINGSIDE_CASTLING_PLANE,
    WHITE_QUEENSIDE_CASTLING_PLANE,
    BoardEncodingError,
    encode_board,
    encode_fen,
    normalized_halfmove_clock,
    square_to_coordinates,
)


def test_board_shape_and_dtype() -> None:
    tensor = encode_board(chess.Board())

    assert tensor.shape == BOARD_SHAPE
    assert tensor.dtype == np.float32


def test_square_to_coordinates() -> None:
    assert square_to_coordinates(chess.A1) == (0, 0)
    assert square_to_coordinates(chess.H1) == (0, 7)
    assert square_to_coordinates(chess.A8) == (7, 0)
    assert square_to_coordinates(chess.H8) == (7, 7)


def test_starting_position_piece_planes() -> None:
    board = chess.Board()
    tensor = encode_board(board)

    white_pawn_plane = PIECE_PLANES[(chess.WHITE, chess.PAWN)]
    black_pawn_plane = PIECE_PLANES[(chess.BLACK, chess.PAWN)]
    white_king_plane = PIECE_PLANES[(chess.WHITE, chess.KING)]
    black_king_plane = PIECE_PLANES[(chess.BLACK, chess.KING)]

    assert tensor[white_pawn_plane].sum() == 8
    assert tensor[black_pawn_plane].sum() == 8
    assert tensor[white_king_plane, 0, 4] == 1.0
    assert tensor[black_king_plane, 7, 4] == 1.0

    assert tensor[:12].sum() == 32


def test_side_to_move_plane() -> None:
    white_board = chess.Board()
    white_tensor = encode_board(white_board)

    assert np.all(white_tensor[SIDE_TO_MOVE_PLANE] == 1.0)

    black_board = chess.Board()
    black_board.push_uci("e2e4")
    black_tensor = encode_board(black_board)

    assert np.all(black_tensor[SIDE_TO_MOVE_PLANE] == 0.0)


def test_castling_right_planes() -> None:
    board = chess.Board()
    tensor = encode_board(board)

    assert np.all(tensor[WHITE_KINGSIDE_CASTLING_PLANE] == 1.0)
    assert np.all(tensor[WHITE_QUEENSIDE_CASTLING_PLANE] == 1.0)
    assert np.all(tensor[BLACK_KINGSIDE_CASTLING_PLANE] == 1.0)
    assert np.all(tensor[BLACK_QUEENSIDE_CASTLING_PLANE] == 1.0)

    board = chess.Board("8/8/8/8/8/8/4K3/7k w - - 0 1")
    tensor = encode_board(board)

    assert tensor[WHITE_KINGSIDE_CASTLING_PLANE].sum() == 0
    assert tensor[WHITE_QUEENSIDE_CASTLING_PLANE].sum() == 0
    assert tensor[BLACK_KINGSIDE_CASTLING_PLANE].sum() == 0
    assert tensor[BLACK_QUEENSIDE_CASTLING_PLANE].sum() == 0


def test_en_passant_plane() -> None:
    board = chess.Board()
    board.push_uci("e2e4")

    tensor = encode_board(board)

    assert tensor[EN_PASSANT_PLANE].sum() == 1.0

    row, column = square_to_coordinates(chess.E3)
    assert tensor[EN_PASSANT_PLANE, row, column] == 1.0


def test_halfmove_clock_plane() -> None:
    board = chess.Board("8/8/8/8/8/8/4K3/7k w - - 37 1")
    tensor = encode_board(board)

    expected = 0.37

    assert normalized_halfmove_clock(board) == expected
    assert np.allclose(tensor[HALFMOVE_CLOCK_PLANE], expected)


def test_halfmove_clock_is_clipped() -> None:
    board = chess.Board("8/8/8/8/8/8/4K3/7k w - - 150 1")

    assert normalized_halfmove_clock(board) == 1.0


def test_encode_fen_matches_encode_board() -> None:
    board = chess.Board()
    from_fen = encode_fen(board.fen())
    from_board = encode_board(board)

    np.testing.assert_array_equal(from_fen, from_board)


def test_encode_fen_rejects_invalid_fen() -> None:
    with pytest.raises(BoardEncodingError, match="Invalid FEN"):
        encode_fen("not a valid fen")
