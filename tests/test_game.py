"""Tests for interactive Yeafins game utilities."""

from pathlib import Path

import chess

from yeafins.engine.game import (
    GameConfig,
    create_pgn,
    format_board,
    parse_move,
    result_description,
    save_pgn,
)


def test_parse_move_accepts_san() -> None:
    board = chess.Board()

    move = parse_move(board, "e4")

    assert move == chess.Move.from_uci("e2e4")


def test_parse_move_accepts_uci() -> None:
    board = chess.Board()

    move = parse_move(board, "e2e4")

    assert move == chess.Move.from_uci("e2e4")


def test_parse_move_rejects_illegal_move() -> None:
    board = chess.Board()

    try:
        parse_move(board, "e2e5")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected illegal move to raise ValueError")


def test_format_board_contains_coordinates() -> None:
    rendered = format_board(chess.Board())

    assert "8  r n b q k b n r" in rendered
    assert "a b c d e f g h" in rendered


def test_result_description_for_checkmate() -> None:
    board = chess.Board()

    for move in (
        "f2f3",
        "e7e5",
        "g2g4",
        "d8h4",
    ):
        board.push_uci(move)

    assert result_description(board) == ("Black by checkmate")


def test_create_pgn_assigns_players() -> None:
    board = chess.Board()
    board.push_uci("e2e4")

    game = create_pgn(
        board,
        player_color="white",
    )

    assert game.headers["White"] == "Krishiv"
    assert game.headers["Black"] == "Yeafins"


def test_save_pgn_creates_file(
    tmp_path: Path,
) -> None:
    board = chess.Board()
    board.push_uci("e2e4")

    output_path = save_pgn(
        board,
        player_color="white",
        output_directory=tmp_path,
    )

    assert output_path.exists()
    assert '[White "Krishiv"]' in (output_path.read_text(encoding="utf-8"))


def test_game_config_defaults() -> None:
    config = GameConfig(checkpoint_path=Path("model.pt"))

    assert config.top_k == 8
    assert config.mode == "blended"
    assert config.depth == 10
    assert config.style_weight is None
