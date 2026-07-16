"""Tests for the Chess.com archive downloader."""

from pathlib import Path

import httpx
import pytest

from yeafins.data.download import (
    DownloadError,
    archive_output_path,
    archive_pgn_url,
    download_archive,
    fetch_archive_urls,
    validate_archive_url,
    validate_pgn_content,
)


def make_client(handler: httpx.MockTransport) -> httpx.Client:
    """Create a test client backed by a mock HTTP transport."""
    return httpx.Client(transport=handler)


def test_validate_archive_url_accepts_expected_url() -> None:
    validate_archive_url(
        "https://api.chess.com/pub/player/yeafins/games/2026/07",
        "yeafins",
    )


@pytest.mark.parametrize(
    "archive_url",
    [
        "http://api.chess.com/pub/player/yeafins/games/2026/07",
        "https://example.com/pub/player/yeafins/games/2026/07",
        "https://api.chess.com/pub/player/someoneelse/games/2026/07",
        "https://api.chess.com/pub/player/yeafins/games/26/07",
        "https://api.chess.com/pub/player/yeafins/games/2026/13",
    ],
)
def test_validate_archive_url_rejects_invalid_urls(archive_url: str) -> None:
    with pytest.raises(DownloadError):
        validate_archive_url(archive_url, "yeafins")


def test_archive_output_path() -> None:
    result = archive_output_path(
        "https://api.chess.com/pub/player/yeafins/games/2026/07",
        Path("data/raw/pgn"),
    )

    assert result == Path("data/raw/pgn/2026-07.pgn")


def test_archive_pgn_url() -> None:
    result = archive_pgn_url("https://api.chess.com/pub/player/yeafins/games/2026/07")

    assert result == ("https://api.chess.com/pub/player/yeafins/games/2026/07/pgn")


def test_validate_pgn_content_accepts_minimal_pgn() -> None:
    content = """
[Event "Live Chess"]
[White "Yeafins"]
[Black "Opponent"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0
"""

    validate_pgn_content(content, "https://example.test/archive")


@pytest.mark.parametrize(
    "content",
    [
        "",
        "   ",
        "<!DOCTYPE html><html></html>",
        '{"games": []}',
        "This is not PGN content.",
    ],
)
def test_validate_pgn_content_rejects_invalid_content(content: str) -> None:
    with pytest.raises(DownloadError):
        validate_pgn_content(content, "https://example.test/archive")


def test_fetch_archive_urls() -> None:
    archive_url = "https://api.chess.com/pub/player/yeafins/games/2026/07"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/pub/player/yeafins/games/archives"
        return httpx.Response(
            200,
            json={"archives": [archive_url]},
        )

    with make_client(httpx.MockTransport(handler)) as client:
        result = fetch_archive_urls(client, "Yeafins")

    assert result == [archive_url]


def test_fetch_archive_urls_rejects_invalid_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": []})

    with make_client(httpx.MockTransport(handler)) as client:
        with pytest.raises(DownloadError):
            fetch_archive_urls(client, "yeafins")


def test_download_archive_writes_pgn(tmp_path: Path) -> None:
    archive_url = "https://api.chess.com/pub/player/yeafins/games/2026/07"
    content = """
[Event "Live Chess"]
[White "Yeafins"]
[Black "Opponent"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0
""".lstrip()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/2026/07/pgn")
        return httpx.Response(200, text=content)

    with make_client(httpx.MockTransport(handler)) as client:
        result = download_archive(
            client,
            archive_url,
            tmp_path,
        )

    output_path = tmp_path / "2026-07.pgn"

    assert result.status == "downloaded"
    assert result.output_path == output_path
    assert output_path.read_text(encoding="utf-8") == content


def test_download_archive_uses_cached_file(tmp_path: Path) -> None:
    archive_url = "https://api.chess.com/pub/player/yeafins/games/2026/07"
    output_path = tmp_path / "2026-07.pgn"
    output_path.write_text("cached", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("No HTTP request should occur for a cached file")

    with make_client(httpx.MockTransport(handler)) as client:
        result = download_archive(
            client,
            archive_url,
            tmp_path,
        )

    assert result.status == "cached"
    assert result.byte_count == len("cached")
