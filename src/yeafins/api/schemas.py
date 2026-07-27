"""Pydantic request and response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ApiMode = Literal["best_of_top_k", "blended"]


class MoveRequest(BaseModel):
    """Parameters for one stateless move decision."""

    model_config = ConfigDict(extra="forbid")

    fen: str = Field(min_length=1)
    top_k: int = Field(default=16, ge=1, le=32)
    mode: ApiMode = "blended"
    stockfish_elo: int = Field(default=2000, ge=1320, le=3190)
    depth: int | None = Field(default=10, ge=1, le=20)
    time_limit_seconds: float | None = Field(default=None, gt=0, le=5)
    style_weight: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def require_analysis_limit(self) -> MoveRequest:
        if self.depth is None and self.time_limit_seconds is None:
            raise ValueError("Either depth or time_limit_seconds must be provided")
        return self


class CandidateResponse(BaseModel):
    move_uci: str
    move_san: str
    model_rank: int
    model_probability: float
    stockfish_cp: int
    selected: bool


class OutcomeResponse(BaseModel):
    result: str
    winner: Literal["white", "black"] | None
    termination: str


class MoveResponse(BaseModel):
    fen: str
    selected_move_uci: str
    selected_move_san: str
    phase: Literal["opening", "middlegame", "endgame"]
    resolved_style_weight: float
    mode: ApiMode
    top_k: int
    stockfish_elo: int
    candidates: list[CandidateResponse]
    game_over: bool
    outcome: OutcomeResponse | None


class HealthResponse(BaseModel):
    status: Literal["ok", "unhealthy"]
    model_loaded: bool
    stockfish_available: bool
