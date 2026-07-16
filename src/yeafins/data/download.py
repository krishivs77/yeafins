"""Download a player's public Chess.com game archives as monthly PGN files."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

LOGGER = logging.getLogger(__name__)

DEFAULT_USERNAME = "yeafins"
DEFAULT_OUTPUT_DIR = Path("data/raw/pgn")
DEFAULT_ARCHIVE_INDEX = Path("data/raw/archives.json")
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_RETRY_DELAY_SECONDS = 1.0

BASE_API_URL = "https://api.chess.com/pub/player"
USER_AGENT = (
    "YeafinsChessResearch/0.1 (personal chess-model research; contact via github repository)"
)


class DownloadError(RuntimeError):
    """Raised when a Chess.com archive cannot be downloaded or validated."""


@dataclass(frozen=True)
class DownloadResult:
    """Summary of a completed archive download."""

    archive_url: str
    output_path: Path
    status: str
    byte_count: int


def create_client() -> httpx.Client:
    """Create the HTTP client used for all Chess.com PubAPI requests."""
    return httpx.Client(
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json, application/x-chess-pgn, text/plain",
        },
        follow_redirects=True,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )


def request_with_retries(
    client: httpx.Client,
    url: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> httpx.Response:
    """Perform an HTTP GET request with exponential-backoff retries."""
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = client.get(url)

            if response.status_code == 200:
                return response

            if response.status_code == 404:
                raise DownloadError(f"Resource not found: {url}")

            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = retry_delay_seconds * (2**attempt)
                else:
                    delay = retry_delay_seconds * (2**attempt)

                LOGGER.warning(
                    "Chess.com returned HTTP %s for %s; retrying in %.1f seconds",
                    response.status_code,
                    url,
                    delay,
                )
                time.sleep(delay)
                continue

            raise DownloadError(
                f"Unexpected HTTP {response.status_code} for {url}: {response.text[:200]!r}"
            )

        except httpx.RequestError as exc:
            last_error = exc

            if attempt == max_retries:
                break

            delay = retry_delay_seconds * (2**attempt)
            LOGGER.warning(
                "Request failed for %s: %s; retrying in %.1f seconds",
                url,
                exc,
                delay,
            )
            time.sleep(delay)

    raise DownloadError(
        f"Failed to retrieve {url} after {max_retries + 1} attempts"
    ) from last_error


def fetch_archive_urls(
    client: httpx.Client,
    username: str,
) -> list[str]:
    """Retrieve the ordered monthly archive URLs for a Chess.com user."""
    normalized_username = username.strip().lower()

    if not normalized_username:
        raise ValueError("Username cannot be empty")

    url = f"{BASE_API_URL}/{normalized_username}/games/archives"
    response = request_with_retries(client, url)

    try:
        payload: Any = response.json()
    except ValueError as exc:
        raise DownloadError(f"Archive index was not valid JSON: {url}") from exc

    if not isinstance(payload, dict):
        raise DownloadError("Archive index must contain a JSON object")

    archives = payload.get("archives")

    if not isinstance(archives, list):
        raise DownloadError("Archive index is missing an 'archives' list")

    validated_archives: list[str] = []

    for archive in archives:
        if not isinstance(archive, str):
            raise DownloadError("Archive index contains a non-string URL")

        validate_archive_url(archive, normalized_username)
        validated_archives.append(archive)

    return validated_archives


def validate_archive_url(archive_url: str, username: str) -> None:
    """Validate that an archive URL belongs to the requested Chess.com user."""
    parsed = urlparse(archive_url)
    path_parts = parsed.path.strip("/").split("/")

    expected_prefix = ["pub", "player", username.lower(), "games"]

    if parsed.scheme != "https" or parsed.netloc.lower() != "api.chess.com":
        raise DownloadError(f"Unexpected archive host: {archive_url}")

    if len(path_parts) != 6:
        raise DownloadError(f"Unexpected archive path: {archive_url}")

    if [part.lower() for part in path_parts[:4]] != expected_prefix:
        raise DownloadError(f"Archive does not belong to {username}: {archive_url}")

    year, month = path_parts[4], path_parts[5]

    if len(year) != 4 or not year.isdigit():
        raise DownloadError(f"Invalid archive year in {archive_url}")

    if len(month) != 2 or not month.isdigit() or not 1 <= int(month) <= 12:
        raise DownloadError(f"Invalid archive month in {archive_url}")


def archive_output_path(archive_url: str, output_dir: Path) -> Path:
    """Convert a monthly archive URL into a deterministic local PGN path."""
    path_parts = urlparse(archive_url).path.strip("/").split("/")
    year, month = path_parts[-2], path_parts[-1]
    return output_dir / f"{year}-{month}.pgn"


def archive_pgn_url(archive_url: str) -> str:
    """Return the PGN endpoint corresponding to a monthly archive URL."""
    return f"{archive_url.rstrip('/')}/pgn"


def validate_pgn_content(content: str, archive_url: str) -> None:
    """Perform inexpensive validation before saving a downloaded PGN."""
    stripped = content.strip()

    if not stripped:
        raise DownloadError(f"Downloaded an empty PGN from {archive_url}")

    lowered = stripped.lower()

    if lowered.startswith("<!doctype html") or lowered.startswith("<html"):
        raise DownloadError(f"Received HTML instead of PGN from {archive_url}")

    required_markers = ("[Event ", "[White ", "[Black ", "[Result ")

    if not all(marker in stripped for marker in required_markers):
        raise DownloadError(f"Downloaded content does not resemble a Chess.com PGN: {archive_url}")


def download_archive(
    client: httpx.Client,
    archive_url: str,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> DownloadResult:
    """Download one monthly archive as PGN."""
    output_path = archive_output_path(archive_url, output_dir)

    if output_path.exists() and not overwrite:
        return DownloadResult(
            archive_url=archive_url,
            output_path=output_path,
            status="cached",
            byte_count=output_path.stat().st_size,
        )

    pgn_url = archive_pgn_url(archive_url)
    response = request_with_retries(client, pgn_url)
    content = response.text

    validate_pgn_content(content, archive_url)

    output_dir.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(".pgn.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(output_path)

    return DownloadResult(
        archive_url=archive_url,
        output_path=output_path,
        status="downloaded",
        byte_count=output_path.stat().st_size,
    )


def write_archive_index(archive_urls: list[str], output_path: Path) -> None:
    """Save the archive URL list as formatted JSON."""
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"archives": archive_urls}
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def download_all_archives(
    username: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    archive_index_path: Path = DEFAULT_ARCHIVE_INDEX,
    *,
    overwrite: bool = False,
) -> list[DownloadResult]:
    """Retrieve the archive index and download every monthly PGN."""
    with create_client() as client:
        archive_urls = fetch_archive_urls(client, username)
        write_archive_index(archive_urls, archive_index_path)

        LOGGER.info("Found %d monthly archives for %s", len(archive_urls), username)

        results: list[DownloadResult] = []

        for index, archive_url in enumerate(archive_urls, start=1):
            LOGGER.info(
                "[%d/%d] Processing %s",
                index,
                len(archive_urls),
                archive_url,
            )

            result = download_archive(
                client,
                archive_url,
                output_dir,
                overwrite=overwrite,
            )
            results.append(result)

            LOGGER.info(
                "%s %s (%d bytes)",
                result.status.capitalize(),
                result.output_path,
                result.byte_count,
            )

        return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download all public monthly Chess.com PGN archives."
    )
    parser.add_argument(
        "--username",
        default=DEFAULT_USERNAME,
        help=f"Chess.com username. Default: {DEFAULT_USERNAME}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for downloaded PGNs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--archive-index",
        type=Path,
        default=DEFAULT_ARCHIVE_INDEX,
        help=f"Path for the archive index JSON. Default: {DEFAULT_ARCHIVE_INDEX}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download archives again even if their PGN files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the archive downloader CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = parse_args()

    results = download_all_archives(
        username=args.username,
        output_dir=args.output_dir,
        archive_index_path=args.archive_index,
        overwrite=args.overwrite,
    )

    downloaded = sum(result.status == "downloaded" for result in results)
    cached = sum(result.status == "cached" for result in results)
    total_bytes = sum(result.byte_count for result in results)

    print()
    print(f"Archives found:      {len(results)}")
    print(f"Archives downloaded: {downloaded}")
    print(f"Archives cached:     {cached}")
    print(f"Total local size:    {total_bytes / (1024 * 1024):.2f} MiB")


if __name__ == "__main__":
    main()
