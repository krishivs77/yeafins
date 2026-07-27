"""Tests for the stateless FastAPI engine interface."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import chess
import pytest
from fastapi.testclient import TestClient

from yeafins.api.app import create_app
from yeafins.api.config import ApiSettings
from yeafins.api.errors import ServiceUnavailableError
from yeafins.api.schemas import (
    CandidateResponse,
    HealthResponse,
    MoveRequest,
    MoveResponse,
)
from yeafins.api.service import YeafinsService
from yeafins.engine.hybrid import CandidateMove, HybridDecision

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class FakeService:
    def __init__(self, *, healthy: bool = True, failure: Exception | None = None) -> None:
        self.healthy = healthy
        self.failure = failure
        self.requests: list[MoveRequest] = []

    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok" if self.healthy else "unhealthy",
            model_loaded=self.healthy,
            stockfish_available=self.healthy,
        )

    async def choose_move(self, request: MoveRequest) -> MoveResponse:
        if self.failure is not None:
            raise self.failure
        self.requests.append(request)
        weight = 0.20 if request.style_weight is None else request.style_weight
        return MoveResponse(
            fen=request.fen,
            selected_move_uci="e2e4",
            selected_move_san="e4",
            phase="opening",
            resolved_style_weight=weight,
            mode=request.mode,
            top_k=request.top_k,
            stockfish_elo=request.stockfish_elo,
            candidates=[
                CandidateResponse(
                    move_uci="e2e4",
                    move_san="e4",
                    model_rank=1,
                    model_probability=0.42,
                    stockfish_cp=31,
                    selected=True,
                ),
                CandidateResponse(
                    move_uci="d2d4",
                    move_san="d4",
                    model_rank=2,
                    model_probability=0.30,
                    stockfish_cp=35,
                    selected=False,
                ),
            ],
            game_over=False,
            outcome=None,
        )


@pytest.fixture
def settings() -> ApiSettings:
    return ApiSettings(
        checkpoint_path=Path("unused.pt"),
        stockfish_path=None,
        allowed_origins=("https://example.vercel.app",),
        stockfish_threads=1,
        stockfish_hash_mb=16,
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )


def test_health_success(settings: ApiSettings) -> None:
    with TestClient(create_app(settings, service=FakeService())) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "stockfish_available": True,
    }


def test_health_unhealthy(settings: ApiSettings) -> None:
    with TestClient(create_app(settings, service=FakeService(healthy=False))) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"


def test_valid_move_serializes_candidates_and_selection(settings: ApiSettings) -> None:
    with TestClient(create_app(settings, service=FakeService())) as client:
        response = client.post("/move", json={"fen": START_FEN})
    body = response.json()
    assert response.status_code == 200
    assert body["selected_move_uci"] == "e2e4"
    assert body["selected_move_san"] == "e4"
    assert body["resolved_style_weight"] == 0.20
    assert body["candidates"][0]["selected"] is True
    assert body["candidates"][1]["move_san"] == "d4"


def test_explicit_style_weight_override(settings: ApiSettings) -> None:
    with TestClient(create_app(settings, service=FakeService())) as client:
        response = client.post("/move", json={"fen": START_FEN, "style_weight": 0.7})
    assert response.json()["resolved_style_weight"] == 0.7


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_k", 0),
        ("top_k", 33),
        ("stockfish_elo", 1319),
        ("stockfish_elo", 3191),
        ("depth", 0),
        ("depth", 21),
        ("time_limit_seconds", 0),
        ("time_limit_seconds", 5.1),
        ("mode", "random"),
        ("style_weight", 1.1),
    ],
)
def test_invalid_parameters_return_structured_422(
    settings: ApiSettings, field: str, value: object
) -> None:
    with TestClient(create_app(settings, service=FakeService())) as client:
        response = client.post("/move", json={"fen": START_FEN, field: value})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_missing_both_analysis_limits_is_invalid(settings: ApiSettings) -> None:
    with TestClient(create_app(settings, service=FakeService())) as client:
        response = client.post(
            "/move", json={"fen": START_FEN, "depth": None, "time_limit_seconds": None}
        )
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("fen", "code"),
    [
        ("not a fen", "invalid_fen"),
        ("7k/5Q2/7K/8/8/8/8/8 b - - 0 1", "terminal_position"),
    ],
)
def test_invalid_chess_positions_return_structured_400(
    settings: ApiSettings, fen: str, code: str
) -> None:
    service = YeafinsService(
        Path("missing-for-test.pt"),
        stockfish_path=None,
        threads=1,
        hash_mb=16,
    )
    with TestClient(create_app(settings, service=service)) as client:
        response = client.post("/move", json={"fen": fen})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == code


def test_structured_service_unavailable(settings: ApiSettings) -> None:
    service = FakeService(failure=ServiceUnavailableError())
    with TestClient(create_app(settings, service=service)) as client:
        response = client.post("/move", json={"fen": START_FEN})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"


def test_structured_unexpected_error(settings: ApiSettings) -> None:
    service = FakeService(failure=RuntimeError("private detail"))
    with TestClient(create_app(settings, service=service), raise_server_exceptions=False) as client:
        response = client.post("/move", json={"fen": START_FEN})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "private detail" not in response.text


def test_cors_allows_only_configured_origin(settings: ApiSettings) -> None:
    app = create_app(settings, service=FakeService())
    with TestClient(app) as client:
        allowed = client.options(
            "/move",
            headers={
                "Origin": "https://example.vercel.app",
                "Access-Control-Request-Method": "POST",
            },
        )
        denied = client.options(
            "/move",
            headers={
                "Origin": "https://attacker.example",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert allowed.headers["access-control-allow-origin"] == "https://example.vercel.app"
    assert "access-control-allow-origin" not in denied.headers


def test_render_safe_stockfish_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STOCKFISH_THREADS", raising=False)
    monkeypatch.delenv("STOCKFISH_HASH_MB", raising=False)
    resolved = ApiSettings.from_env()
    assert resolved.stockfish_threads == 1
    assert resolved.stockfish_hash_mb == 16


def test_service_startup_initializes_engine_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")
    engine = MagicMock()
    constructor = MagicMock(return_value=engine)
    monkeypatch.setattr("yeafins.api.service.YeafinsHybridEngine", constructor)
    service = YeafinsService(
        checkpoint,
        stockfish_path="stockfish",
        threads=1,
        hash_mb=16,
    )

    asyncio.run(service.startup())
    asyncio.run(service.startup())
    assert constructor.call_count == 1
    assert service.health().status == "ok"
    asyncio.run(service.shutdown())


def test_service_preserves_model_rank_candidate_order_and_contract() -> None:
    service = YeafinsService(
        Path("unused.pt"),
        stockfish_path=None,
        threads=1,
        hash_mb=16,
    )
    engine = MagicMock()
    e4 = chess.Move.from_uci("e2e4")
    d4 = chess.Move.from_uci("d2d4")
    nf3 = chess.Move.from_uci("g1f3")
    engine.choose_move.return_value = HybridDecision(
        selected_move=d4,
        candidates=(
            CandidateMove(nf3, 0.2, 3, 10),
            CandidateMove(e4, 0.5, 1, 20),
            CandidateMove(d4, 0.3, 2, 30),
        ),
        mode="blended",
        top_k=16,
    )
    service._engine = engine
    request = MoveRequest(
        fen=START_FEN,
        mode="blended",
        top_k=16,
        stockfish_elo=2000,
        depth=None,
        time_limit_seconds=1.5,
        style_weight=None,
    )

    response = service._choose_move_sync(chess.Board(), request)

    assert [candidate.model_rank for candidate in response.candidates] == [1, 2, 3]
    assert [candidate.move_uci for candidate in response.candidates] == [
        "e2e4",
        "d2d4",
        "g1f3",
    ]
    assert response.model_dump().keys() == {
        "fen",
        "selected_move_uci",
        "selected_move_san",
        "phase",
        "resolved_style_weight",
        "mode",
        "top_k",
        "stockfish_elo",
        "candidates",
        "game_over",
        "outcome",
    }
    engine.engine.configure.assert_called_once_with({"UCI_LimitStrength": True, "UCI_Elo": 2000})
    engine.choose_move.assert_called_once_with(
        chess.Board(),
        top_k=16,
        mode="blended",
        depth=None,
        time_limit_seconds=1.5,
        style_weight=0.20,
    )
