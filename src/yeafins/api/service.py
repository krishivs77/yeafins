"""Application-wide model and Stockfish resource coordination."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Protocol

import chess
import chess.engine

from yeafins.api.errors import ApiError, ServiceUnavailableError
from yeafins.api.schemas import (
    CandidateResponse,
    HealthResponse,
    MoveRequest,
    MoveResponse,
)
from yeafins.engine.hybrid import (
    HybridEngineError,
    YeafinsHybridEngine,
    infer_game_phase,
    phase_style_weight,
)

LOGGER = logging.getLogger(__name__)


class MoveService(Protocol):
    async def startup(self) -> None: ...
    async def shutdown(self) -> None: ...
    def health(self) -> HealthResponse: ...
    async def choose_move(self, request: MoveRequest) -> MoveResponse: ...


class YeafinsService:
    """Own one loaded model and one serialized Stockfish process."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        stockfish_path: str | None,
        threads: int,
        hash_mb: int,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.stockfish_path = stockfish_path
        self.threads = threads
        self.hash_mb = hash_mb
        self._engine: YeafinsHybridEngine | None = None
        self._lock = asyncio.Lock()
        self._model_loaded = False
        self._stockfish_available = False

    async def startup(self) -> None:
        if not self.checkpoint_path.is_file():
            LOGGER.error("Configured Yeafins checkpoint does not exist")
            return
        try:
            self._engine = await asyncio.to_thread(
                YeafinsHybridEngine,
                self.checkpoint_path,
                stockfish_path=self.stockfish_path,
                threads=self.threads,
                hash_mb=self.hash_mb,
            )
        except Exception:
            LOGGER.exception("Failed to initialize the Yeafins inference service")
            return
        self._model_loaded = True
        self._stockfish_available = True

    async def shutdown(self) -> None:
        engine, self._engine = self._engine, None
        self._stockfish_available = False
        self._model_loaded = False
        if engine is not None:
            try:
                await asyncio.to_thread(engine.close)
            except Exception:
                LOGGER.exception("Failed to shut down Stockfish cleanly")

    def health(self) -> HealthResponse:
        ready = self._engine is not None and self._model_loaded and self._stockfish_available
        return HealthResponse(
            status="ok" if ready else "unhealthy",
            model_loaded=self._model_loaded,
            stockfish_available=self._stockfish_available,
        )

    async def choose_move(self, request: MoveRequest) -> MoveResponse:
        try:
            board = chess.Board(request.fen)
        except ValueError as error:
            raise ApiError(400, "invalid_fen", "The supplied FEN is invalid.") from error

        if board.is_game_over(claim_draw=True):
            raise ApiError(400, "terminal_position", "The supplied position is already game over.")
        if board.legal_moves.count() == 0:
            raise ApiError(400, "no_legal_moves", "The supplied position has no legal moves.")

        async with self._lock:
            if self._engine is None or not self._stockfish_available:
                raise ServiceUnavailableError()
            try:
                response = await asyncio.to_thread(self._choose_move_sync, board, request)
            except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as error:
                self._stockfish_available = False
                LOGGER.exception("Stockfish failed while serving a move request")
                raise ServiceUnavailableError("Stockfish is currently unavailable.") from error
            except HybridEngineError as error:
                LOGGER.warning("Hybrid engine could not produce a move: %s", error)
                raise ServiceUnavailableError(
                    "The chess engine could not produce a move."
                ) from error
            return response

    def _choose_move_sync(self, board: chess.Board, request: MoveRequest) -> MoveResponse:
        assert self._engine is not None
        self._engine.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": request.stockfish_elo})
        resolved_weight = (
            phase_style_weight(board) if request.style_weight is None else request.style_weight
        )
        decision = self._engine.choose_move(
            board,
            top_k=request.top_k,
            mode=request.mode,
            depth=request.depth,
            time_limit_seconds=request.time_limit_seconds,
            style_weight=resolved_weight,
        )
        selected = decision.selected_move
        candidates = [
            CandidateResponse(
                move_uci=candidate.move.uci(),
                move_san=board.san(candidate.move),
                model_rank=candidate.model_rank,
                model_probability=candidate.model_probability,
                stockfish_cp=candidate.stockfish_cp,
                selected=candidate.move == selected,
            )
            for candidate in sorted(decision.candidates, key=lambda item: item.model_rank)
        ]
        return MoveResponse(
            fen=request.fen,
            selected_move_uci=selected.uci(),
            selected_move_san=board.san(selected),
            phase=infer_game_phase(board),
            resolved_style_weight=resolved_weight,
            mode=request.mode,
            top_k=request.top_k,
            stockfish_elo=request.stockfish_elo,
            candidates=candidates,
            game_over=False,
            outcome=None,
        )
