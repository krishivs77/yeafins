"""Tests for PGN corpus parsing."""

from io import StringIO
from pathlib import Path

import chess.pgn

from yeafins.data.parse import (
    ParsedGame,
    RejectedGame,
    infer_player_result,
    infer_time_class,
    parse_game,
    parse_optional_bool,
    parse_optional_int,
)


def read_single_game(pgn: str) -> chess.pgn.Game:
    """Read one game from an in-memory PGN."""
    game = chess.pgn.read_game(StringIO(pgn))
    assert game is not None
    return game


def test_parse_optional_int() -> None:
    assert parse_optional_int("1542") == 1542
    assert parse_optional_int("?") is None
    assert parse_optional_int("unknown") is None
    assert parse_optional_int(None) is None


def test_parse_optional_bool() -> None:
    assert parse_optional_bool("True") is True
    assert parse_optional_bool("false") is False
    assert parse_optional_bool("unknown") is None
    assert parse_optional_bool(None) is None


def test_infer_player_result() -> None:
    assert infer_player_result("1-0", "white") == "win"
    assert infer_player_result("1-0", "black") == "loss"
    assert infer_player_result("0-1", "black") == "win"
    assert infer_player_result("1/2-1/2", "white") == "draw"
    assert infer_player_result("*", "white") == "unknown"


def test_infer_time_class_from_explicit_header() -> None:
    headers = chess.pgn.Headers()
    headers["TimeClass"] = "Rapid"
    headers["TimeControl"] = "180"

    assert infer_time_class(headers) == "rapid"


def test_infer_time_class_from_time_control() -> None:
    expected = {
        "30": "bullet",
        "60": "bullet",
        "60+1": "bullet",
        "120+1": "blitz",
        "180": "blitz",
        "180+2": "blitz",
        "300": "blitz",
        "300+5": "blitz",
        "600": "rapid",
        "900+10": "rapid",
        "1800": "rapid",
        "3600": "rapid",
        "1/259200": "daily",
    }

    for time_control, time_class in expected.items():
        headers = chess.pgn.Headers()
        headers["TimeControl"] = time_control

        assert infer_time_class(headers) == time_class


def test_infer_time_class_returns_unknown_for_unrecognized_control() -> None:
    headers = chess.pgn.Headers()
    headers["TimeControl"] = "45+5"

    assert infer_time_class(headers) == "unknown"


def test_parse_standard_game_as_white() -> None:
    game = read_single_game(
        """
[Event "Live Chess"]
[Site "https://www.chess.com/game/live/123"]
[Date "2026.07.16"]
[UTCDate "2026.07.16"]
[UTCTime "14:00:00"]
[White "Yeafins"]
[Black "Opponent"]
[WhiteElo "1500"]
[BlackElo "1520"]
[TimeControl "600"]
[TimeClass "rapid"]
[Result "1-0"]
[Termination "Yeafins won by checkmate"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6
5. O-O Be7 6. Re1 b5 7. Bb3 O-O 8. c3 d6
9. h3 Nb8 10. d4 Nbd7 11. Ng5 Bb7 12. Bxf7+ Rxf7
13. Ne6 Qc8 14. Ng5 Rf8 15. Qb3+ d5 16. exd5 Bxd5
17. Qc2 h6 18. dxe5 hxg5 19. exf6 Bxf6 20. Be3 1-0
"""
    )

    result = parse_game(
        game,
        username="yeafins",
        source_file=Path("2026-07.pgn"),
        source_game_index=1,
    )

    assert isinstance(result, ParsedGame)
    assert result.player_color == "white"
    assert result.player_rating == 1500
    assert result.opponent_rating == 1520
    assert result.player_result == "win"
    assert result.time_class == "rapid"
    assert result.moves_uci[0] == "e2e4"
    assert result.moves_uci[1] == "e7e5"
    assert result.ply_count == len(result.moves_uci)


def test_parse_game_as_black() -> None:
    game = read_single_game(
        """
[Event "Live Chess"]
[White "Opponent"]
[Black "YEAFINS"]
[WhiteElo "1450"]
[BlackElo "1500"]
[Result "0-1"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 0-1
"""
    )

    result = parse_game(
        game,
        username="yeafins",
        source_file=Path("2026-07.pgn"),
        source_game_index=1,
    )

    assert isinstance(result, ParsedGame)
    assert result.player_color == "black"
    assert result.player_rating == 1500
    assert result.player_result == "win"


def test_rejects_variant_game() -> None:
    game = read_single_game(
        """
[Event "Chess960"]
[Variant "Chess960"]
[White "Yeafins"]
[Black "Opponent"]
[Result "1-0"]

1. e4 e5 1-0
"""
    )

    result = parse_game(
        game,
        username="yeafins",
        source_file=Path("variant.pgn"),
        source_game_index=1,
    )

    assert isinstance(result, RejectedGame)
    assert result.reason == "variant:chess960"


def test_rejects_game_without_player() -> None:
    game = read_single_game(
        """
[Event "Live Chess"]
[White "PlayerOne"]
[Black "PlayerTwo"]
[Result "1-0"]

1. e4 e5 1-0
"""
    )

    result = parse_game(
        game,
        username="yeafins",
        source_file=Path("other.pgn"),
        source_game_index=1,
    )

    assert isinstance(result, RejectedGame)
    assert result.reason == "player_not_found"


def test_rejects_game_without_moves() -> None:
    game = read_single_game(
        """
[Event "Live Chess"]
[White "Yeafins"]
[Black "Opponent"]
[Result "*"]

*
"""
    )

    result = parse_game(
        game,
        username="yeafins",
        source_file=Path("empty.pgn"),
        source_game_index=1,
    )

    assert isinstance(result, RejectedGame)
    assert result.reason == "no_moves"
