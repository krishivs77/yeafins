"""Encode chess moves into an AlphaZero-style 8 × 8 × 73 policy space."""

from __future__ import annotations

import chess
import numpy as np

POLICY_PLANES = 73
POLICY_SIZE = 64 * POLICY_PLANES

# Absolute board-coordinate directions.
# Files increase toward h; ranks increase toward rank 8.
SLIDING_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),  # north
    (1, 1),  # northeast
    (1, 0),  # east
    (1, -1),  # southeast
    (0, -1),  # south
    (-1, -1),  # southwest
    (-1, 0),  # west
    (-1, 1),  # northwest
)

KNIGHT_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)

UNDERPROMOTION_PIECES: tuple[int, ...] = (
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
)


class MoveEncodingError(ValueError):
    """Raised when a move cannot be represented in the policy space."""


def square_delta(from_square: chess.Square, to_square: chess.Square) -> tuple[int, int]:
    """Return the file and rank displacement between two squares."""
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    to_file = chess.square_file(to_square)
    to_rank = chess.square_rank(to_square)

    return to_file - from_file, to_rank - from_rank


def sliding_plane(file_delta: int, rank_delta: int) -> int | None:
    """Return a sliding-move plane, or None if the move is not queen-like."""
    if file_delta == 0 and rank_delta == 0:
        return None

    distance = max(abs(file_delta), abs(rank_delta))

    is_straight = file_delta == 0 or rank_delta == 0
    is_diagonal = abs(file_delta) == abs(rank_delta)

    if not (is_straight or is_diagonal):
        return None

    direction = (
        0 if file_delta == 0 else file_delta // abs(file_delta),
        0 if rank_delta == 0 else rank_delta // abs(rank_delta),
    )

    try:
        direction_index = SLIDING_DIRECTIONS.index(direction)
    except ValueError:
        return None

    if not 1 <= distance <= 7:
        return None

    return direction_index * 7 + (distance - 1)


def knight_plane(file_delta: int, rank_delta: int) -> int | None:
    """Return a knight-move plane, or None if the move is not knight-like."""
    try:
        direction_index = KNIGHT_DIRECTIONS.index((file_delta, rank_delta))
    except ValueError:
        return None

    return 56 + direction_index


def underpromotion_plane(
    move: chess.Move,
    board: chess.Board,
) -> int | None:
    """Return an underpromotion plane for knight, bishop, or rook promotion."""
    if move.promotion not in UNDERPROMOTION_PIECES:
        return None

    file_delta, rank_delta = square_delta(move.from_square, move.to_square)

    forward_rank_delta = 1 if board.turn == chess.WHITE else -1

    if rank_delta != forward_rank_delta:
        raise MoveEncodingError(f"Invalid promotion displacement for {move.uci()} in {board.fen()}")

    # Relative to the moving player's perspective.
    if board.turn == chess.WHITE:
        relative_file_delta = file_delta
    else:
        relative_file_delta = -file_delta

    promotion_directions = {
        0: 0,  # straight forward
        -1: 1,  # capture left from player's perspective
        1: 2,  # capture right from player's perspective
    }

    if relative_file_delta not in promotion_directions:
        raise MoveEncodingError(f"Invalid underpromotion direction for {move.uci()}")

    piece_index = UNDERPROMOTION_PIECES.index(move.promotion)
    direction_index = promotion_directions[relative_file_delta]

    return 64 + piece_index * 3 + direction_index


def encode_move(move: chess.Move, board: chess.Board) -> int:
    """Encode a legal move into a flattened policy index from 0 to 4671."""
    if move not in board.legal_moves:
        raise MoveEncodingError(
            f"Cannot encode illegal move {move.uci()} in position {board.fen()}"
        )

    plane = underpromotion_plane(move, board)

    if plane is None:
        file_delta, rank_delta = square_delta(move.from_square, move.to_square)

        plane = knight_plane(file_delta, rank_delta)

        if plane is None:
            plane = sliding_plane(file_delta, rank_delta)

    if plane is None:
        raise MoveEncodingError(f"Move {move.uci()} could not be mapped to a policy plane")

    return move.from_square * POLICY_PLANES + plane


def decode_move(
    policy_index: int,
    board: chess.Board,
) -> chess.Move:
    """Decode a policy index into a legal move for the supplied position."""
    if not 0 <= policy_index < POLICY_SIZE:
        raise MoveEncodingError(f"Policy index must be between 0 and {POLICY_SIZE - 1}")

    from_square, plane = divmod(policy_index, POLICY_PLANES)
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)

    promotion: int | None = None

    if plane < 56:
        direction_index, distance_index = divmod(plane, 7)
        distance = distance_index + 1
        file_step, rank_step = SLIDING_DIRECTIONS[direction_index]

        to_file = from_file + file_step * distance
        to_rank = from_rank + rank_step * distance

    elif plane < 64:
        direction_index = plane - 56
        file_step, rank_step = KNIGHT_DIRECTIONS[direction_index]

        to_file = from_file + file_step
        to_rank = from_rank + rank_step

    else:
        underpromotion_index = plane - 64
        piece_index, direction_index = divmod(underpromotion_index, 3)

        promotion = UNDERPROMOTION_PIECES[piece_index]

        relative_file_steps = (0, -1, 1)
        relative_file_step = relative_file_steps[direction_index]

        if board.turn == chess.WHITE:
            file_step = relative_file_step
            rank_step = 1
        else:
            file_step = -relative_file_step
            rank_step = -1

        to_file = from_file + file_step
        to_rank = from_rank + rank_step

    if not 0 <= to_file < 8 or not 0 <= to_rank < 8:
        raise MoveEncodingError(f"Policy index {policy_index} points outside the board")

    to_square = chess.square(to_file, to_rank)

    # Queen promotions use the corresponding sliding-move plane.
    if promotion is None:
        moving_piece = board.piece_at(from_square)

        if moving_piece is not None and moving_piece.piece_type == chess.PAWN and to_rank in {0, 7}:
            promotion = chess.QUEEN

    move = chess.Move(
        from_square=from_square,
        to_square=to_square,
        promotion=promotion,
    )

    if move not in board.legal_moves:
        raise MoveEncodingError(
            f"Policy index {policy_index} decodes to illegal move {move.uci()} in {board.fen()}"
        )

    return move


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Return a Boolean mask over all 4,672 policy outputs."""
    mask = np.zeros(POLICY_SIZE, dtype=np.bool_)

    for move in board.legal_moves:
        mask[encode_move(move, board)] = True

    return mask


def legal_policy_indices(board: chess.Board) -> list[int]:
    """Return encoded indices for every legal move in the position."""
    return [encode_move(move, board) for move in board.legal_moves]
