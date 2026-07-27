"""Environment-backed API configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class ApiSettings:
    """Configuration that may be supplied by the hosting platform."""

    checkpoint_path: Path
    stockfish_path: str | None
    allowed_origins: tuple[str, ...]
    stockfish_threads: int
    stockfish_hash_mb: int
    host: str
    port: int
    log_level: str

    @classmethod
    def from_env(cls) -> ApiSettings:
        origins = tuple(
            origin.strip()
            for origin in os.getenv(
                "ALLOWED_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000",
            ).split(",")
            if origin.strip()
        )
        return cls(
            checkpoint_path=Path(
                os.getenv("YEAFINS_CHECKPOINT_PATH", "runs/resnet_baseline/best.pt")
            ),
            stockfish_path=os.getenv("STOCKFISH_PATH") or None,
            allowed_origins=origins,
            stockfish_threads=_positive_int("STOCKFISH_THREADS", 1),
            stockfish_hash_mb=_positive_int("STOCKFISH_HASH_MB", 16),
            host=os.getenv("HOST", "0.0.0.0"),
            port=_positive_int("PORT", 8000),
            log_level=os.getenv("LOG_LEVEL", "info").lower(),
        )
