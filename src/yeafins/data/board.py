"""Encode chess positions as fixed-shape neural-network input tensors."""

from __future__ import annotations

import chess
import numpy as np

BOARD_PLANES = 19
BOARD_HEIGHT = 8
BOARD_WIDTH = 8
BOARD_SHAPE = (BOARD_PLANES, BOARD_HEIGHT, BOARD_WIDTH)

PIECE_PLANES: dict[tuple[chess.Color, chess.PieceType], int] = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}

SIDE_TO_MOVE_PLANE = 12
WHITE_KINGSIDE_CASTLING_PLANE = 13
WHITE_QUEENSIDE_CASTLING_PLANE = 14
BLACK_KINGSIDE_CASTLING_PLANE = 15
BLACK_QUEENSIDE_CASTLING_PLANE = 16
EN_PASSANT_PLANE = 17
HALFMOVE_CLOCK_PLANE = 18

MAX_HALFMOVE_CLOCK = 100


class BoardEncodingError(ValueError):
    """Raised when a board position cannot be encoded safely."""


def square_to_coordinates(square: chess.Square) -> tuple[int, int]:
    """Convert a chess square into tensor row and column coordinates.

    Tensor rows follow rank order:
    row 0 -> rank 1
    row 7 -> rank 8

    Tensor columns follow file order:
    column 0 -> file a
    column 7 -> file h
    """
    row = chess.square_rank(square)
    column = chess.square_file(square)
    return row, column


def normalized_halfmove_clock(board: chess.Board) -> float:
    """Normalize the fifty-move-rule clock into the range [0, 1]."""
    clipped = min(max(board.halfmove_clock, 0), MAX_HALFMOVE_CLOCK)
    return clipped / MAX_HALFMOVE_CLOCK


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a standard chess board as a float32 tensor."""
    if type(board) is not chess.Board:
        raise BoardEncodingError(f"Expected chess.Board, received {type(board).__name__}")

    if board.chess960:
        raise BoardEncodingError("Chess960 positions are not supported")

    tensor = np.zeros(BOARD_SHAPE, dtype=np.float32)

    for square, piece in board.piece_map().items():
        plane = PIECE_PLANES[(piece.color, piece.piece_type)]
        row, column = square_to_coordinates(square)
        tensor[plane, row, column] = 1.0

    if board.turn == chess.WHITE:
        tensor[SIDE_TO_MOVE_PLANE, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[WHITE_KINGSIDE_CASTLING_PLANE, :, :] = 1.0

    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[WHITE_QUEENSIDE_CASTLING_PLANE, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[BLACK_KINGSIDE_CASTLING_PLANE, :, :] = 1.0

    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[BLACK_QUEENSIDE_CASTLING_PLANE, :, :] = 1.0

    if board.ep_square is not None:
        row, column = square_to_coordinates(board.ep_square)
        tensor[EN_PASSANT_PLANE, row, column] = 1.0

    tensor[HALFMOVE_CLOCK_PLANE, :, :] = normalized_halfmove_clock(board)

    return tensor


def encode_fen(fen: str) -> np.ndarray:
    """Parse and encode a FEN string."""
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise BoardEncodingError(f"Invalid FEN: {fen!r}") from exc

    return encode_board(board)
