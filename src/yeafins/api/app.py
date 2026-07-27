"""FastAPI application factory and command-line entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from yeafins.api.config import ApiSettings
from yeafins.api.errors import ApiError
from yeafins.api.schemas import HealthResponse, MoveRequest, MoveResponse
from yeafins.api.service import MoveService, YeafinsService

LOGGER = logging.getLogger(__name__)


def create_app(
    settings: ApiSettings | None = None,
    *,
    service: MoveService | None = None,
) -> FastAPI:
    """Build an app with injectable resources for isolated tests."""
    resolved = settings or ApiSettings.from_env()
    move_service = service or YeafinsService(
        resolved.checkpoint_path,
        stockfish_path=resolved.stockfish_path,
        threads=resolved.stockfish_threads,
        hash_mb=resolved.stockfish_hash_mb,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.move_service = move_service
        await move_service.startup()
        try:
            yield
        finally:
            await move_service.shutdown()

    app = FastAPI(title="Yeafins Engine API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )

    @app.exception_handler(ApiError)
    async def api_error_handler(_request: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"code": error.code, "message": error.message}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "The request body failed validation.",
                }
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_request: Request, error: Exception) -> JSONResponse:
        LOGGER.exception("Unexpected API error", exc_info=error)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected internal error occurred.",
                }
            },
        )

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> JSONResponse | HealthResponse:
        result = request.app.state.move_service.health()
        if result.status != "ok":
            return JSONResponse(status_code=503, content=result.model_dump())
        return result

    @app.post("/move", response_model=MoveResponse)
    async def move(payload: MoveRequest, request: Request) -> MoveResponse:
        return await request.app.state.move_service.choose_move(payload)

    return app


app = create_app()


def main() -> None:
    """Run the API using environment-backed Uvicorn settings."""
    settings = ApiSettings.from_env()
    uvicorn.run(
        "yeafins.api.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )
