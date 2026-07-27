"""Measure Yeafins API inference RSS with the production engine configuration."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ["YEAFINS_LOG_MEMORY"] = "true"
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

from yeafins.runtime import current_rss_mb  # noqa: E402


def report(label: str) -> None:
    print(f"{label}: {current_rss_mb():.1f} MiB", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/best_inference.pt"),
    )
    parser.add_argument("--stockfish-path")
    parser.add_argument("--requests", type=int, default=5)
    arguments = parser.parse_args()

    report("process start")

    import chess

    from yeafins.api.schemas import MoveRequest
    from yeafins.api.service import YeafinsService

    report("after importing API modules")

    service = YeafinsService(
        arguments.checkpoint,
        stockfish_path=arguments.stockfish_path,
        threads=1,
        hash_mb=16,
    )

    import asyncio

    async def measure() -> None:
        await service.startup()
        report("after loading policy model and starting Stockfish")
        if service._engine is None:
            raise RuntimeError("Engine failed to start; inspect the preceding log")

        board = chess.Board()
        for index in range(arguments.requests):
            response = await service.choose_move(MoveRequest(fen=board.fen()))
            board.push_uci(response.selected_move_uci)
            if not board.is_game_over():
                board.push(next(iter(board.legal_moves)))
            report(f"after move request {index + 1}")
        await service.shutdown()

    asyncio.run(measure())


if __name__ == "__main__":
    main()
