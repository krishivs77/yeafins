"""Combine the personalized policy model with Stockfish evaluation."""

from __future__ import annotations

import gc
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import chess
import chess.engine
import torch

from yeafins.data.board import encode_board
from yeafins.data.encode import decode_move, legal_move_mask
from yeafins.models.resnet_policy import (
    ResNetPolicy,
    ResNetPolicyConfig,
    apply_legal_move_mask,
)
from yeafins.runtime import log_memory, select_device

SelectionMode = Literal["best_of_top_k", "blended"]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateMove:
    """One personalized move candidate with model and engine scores."""

    move: chess.Move
    model_probability: float
    model_rank: int
    stockfish_cp: int
    blended_score: float | None = None


@dataclass(frozen=True)
class HybridDecision:
    """Complete record of a hybrid move-selection decision."""

    selected_move: chess.Move
    candidates: tuple[CandidateMove, ...]
    mode: SelectionMode
    top_k: int


class HybridEngineError(RuntimeError):
    """Raised when the hybrid engine cannot produce a move."""


def load_policy_model(
    checkpoint_path: Path,
) -> tuple[ResNetPolicy, torch.device]:
    """Load a trained policy model and select its inference device."""
    device = select_device()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = ResNetPolicy(ResNetPolicyConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    del checkpoint
    gc.collect()
    log_memory("policy model loaded", logger=LOGGER)

    return model, device


def resolve_stockfish_path(
    stockfish_path: str | None = None,
) -> str:
    """Resolve the Stockfish executable."""
    if stockfish_path is not None:
        return stockfish_path

    resolved = shutil.which("stockfish")

    if resolved is None:
        raise HybridEngineError("Stockfish was not found on PATH")

    return resolved


def model_candidates(
    model: ResNetPolicy,
    board: chess.Board,
    *,
    device: torch.device,
    top_k: int,
    temperature: float = 1.0,
) -> list[tuple[chess.Move, float, int]]:
    """Return the policy model's top legal moves."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    legal_count = board.legal_moves.count()

    if legal_count == 0:
        raise HybridEngineError("Cannot select a move from a terminal position")

    effective_k = min(top_k, legal_count)

    board_tensor = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)

    numpy_mask = legal_move_mask(board)
    legal_mask_tensor = torch.from_numpy(numpy_mask).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(board_tensor)
        masked_logits = apply_legal_move_mask(
            logits / temperature,
            legal_mask_tensor,
        )
        probabilities = torch.softmax(
            masked_logits,
            dim=1,
        )

        top_probabilities, top_indices = probabilities.topk(
            effective_k,
            dim=1,
        )

    candidates: list[tuple[chess.Move, float, int]] = []

    for rank, (policy_index, probability) in enumerate(
        zip(
            top_indices[0].cpu().tolist(),
            top_probabilities[0].cpu().tolist(),
            strict=True,
        ),
        start=1,
    ):
        move = decode_move(int(policy_index), board)

        candidates.append(
            (
                move,
                float(probability),
                rank,
            )
        )

    return candidates


def score_from_player_perspective(
    score: chess.engine.PovScore,
    board: chess.Board,
    *,
    mate_score: int = 100_000,
) -> int:
    """Convert a Stockfish score into centipawns for the side to move."""
    relative = score.pov(board.turn)

    value = relative.score(
        mate_score=mate_score,
    )

    if value is None:
        raise HybridEngineError("Stockfish returned a score without a numeric value")

    return int(value)


def evaluate_root_moves(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    moves: list[chess.Move],
    *,
    depth: int | None,
    time_limit_seconds: float | None,
) -> dict[chess.Move, int]:
    """Evaluate restricted root moves in one MultiPV search."""
    if not moves:
        raise ValueError("At least one root move is required")

    illegal_moves = [move for move in moves if move not in board.legal_moves]
    if illegal_moves:
        rendered = ", ".join(move.uci() for move in illegal_moves)
        raise HybridEngineError(f"Illegal root moves supplied: {rendered}")

    if depth is None and time_limit_seconds is None:
        raise ValueError("Either depth or time_limit_seconds must be provided")

    if time_limit_seconds is not None:
        if time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        limit = chess.engine.Limit(time=time_limit_seconds)
    else:
        if depth is None or depth <= 0:
            raise ValueError("depth must be positive")
        limit = chess.engine.Limit(depth=depth)

    information = engine.analyse(
        board,
        limit,
        multipv=len(moves),
        root_moves=moves,
    )
    results = [information] if isinstance(information, dict) else information
    scores: dict[chess.Move, int] = {}

    for item in results:
        principal_variation = item.get("pv")
        if not principal_variation:
            continue

        root_move = principal_variation[0]
        if root_move not in moves:
            continue

        score = item.get("score")
        if score is None:
            raise HybridEngineError(f"Stockfish returned no score for {root_move.uci()}")
        scores[root_move] = score_from_player_perspective(score, board)

    missing = [move for move in moves if move not in scores]
    if missing:
        rendered = ", ".join(move.uci() for move in missing)
        raise HybridEngineError(f"Stockfish did not score all root moves: {rendered}")

    return scores


def evaluate_candidates(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    candidates: list[tuple[chess.Move, float, int]],
    *,
    depth: int | None = 12,
    time_limit_seconds: float | None = None,
) -> list[CandidateMove]:
    """Evaluate all proposed moves in one restricted MultiPV search."""
    moves = [move for move, _, _ in candidates]
    scores = evaluate_root_moves(
        engine,
        board,
        moves,
        depth=depth,
        time_limit_seconds=time_limit_seconds,
    )

    return [
        CandidateMove(
            move=move,
            model_probability=probability,
            model_rank=rank,
            stockfish_cp=scores[move],
        )
        for move, probability, rank in candidates
    ]


def infer_game_phase(board: chess.Board) -> str:
    """Classify a position as opening, middlegame, or endgame."""
    if board.fullmove_number <= 10:
        return "opening"

    values = {
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    non_pawn_material = sum(
        len(board.pieces(piece_type, color)) * value
        for piece_type, value in values.items()
        for color in (chess.WHITE, chess.BLACK)
    )

    if non_pawn_material <= 14:
        return "endgame"

    return "middlegame"


def phase_style_weight(board: chess.Board) -> float:
    """Return the validated default style weight for the position phase."""
    phase = infer_game_phase(board)

    weights = {
        "opening": 0.20,
        "middlegame": 0.10,
        "endgame": 0.20,
    }

    return weights[phase]


def normalize_stockfish_scores(
    candidates: list[CandidateMove],
) -> dict[chess.Move, float]:
    """Scale candidate Stockfish scores into the range [0, 1]."""
    values = [candidate.stockfish_cp for candidate in candidates]

    minimum = min(values)
    maximum = max(values)

    if minimum == maximum:
        return {candidate.move: 1.0 for candidate in candidates}

    return {
        candidate.move: (candidate.stockfish_cp - minimum) / (maximum - minimum)
        for candidate in candidates
    }


def choose_best_of_top_k(
    candidates: list[CandidateMove],
) -> CandidateMove:
    """Choose the strongest Stockfish move among model candidates."""
    if not candidates:
        raise HybridEngineError("No candidates were provided")

    return max(
        candidates,
        key=lambda candidate: (
            candidate.stockfish_cp,
            candidate.model_probability,
            -candidate.model_rank,
        ),
    )


def choose_blended(
    candidates: list[CandidateMove],
    *,
    style_weight: float = 0.65,
) -> CandidateMove:
    """Choose using normalized model preference and engine quality."""
    if not candidates:
        raise HybridEngineError("No candidates were provided")

    if not 0.0 <= style_weight <= 1.0:
        raise ValueError("style_weight must be between 0 and 1")

    engine_scores = normalize_stockfish_scores(candidates)

    probabilities = [candidate.model_probability for candidate in candidates]
    minimum_probability = min(probabilities)
    maximum_probability = max(probabilities)

    if minimum_probability == maximum_probability:
        model_scores = {candidate.move: 1.0 for candidate in candidates}
    else:
        model_scores = {
            candidate.move: (candidate.model_probability - minimum_probability)
            / (maximum_probability - minimum_probability)
            for candidate in candidates
        }

    rescored: list[CandidateMove] = []

    for candidate in candidates:
        blended_score = (
            style_weight * model_scores[candidate.move]
            + (1.0 - style_weight) * engine_scores[candidate.move]
        )

        rescored.append(
            CandidateMove(
                move=candidate.move,
                model_probability=candidate.model_probability,
                model_rank=candidate.model_rank,
                stockfish_cp=candidate.stockfish_cp,
                blended_score=blended_score,
            )
        )

    return max(
        rescored,
        key=lambda candidate: (
            float(candidate.blended_score),
            candidate.stockfish_cp,
            candidate.model_probability,
        ),
    )


def validate_stockfish_elo(stockfish_elo: int) -> None:
    """Validate a Stockfish limited-strength Elo setting."""
    if not 1320 <= stockfish_elo <= 3190:
        raise ValueError("stockfish_elo must be between 1320 and 3190")


class YeafinsHybridEngine:
    """Personalized move proposer with Stockfish verification."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        stockfish_path: str | None = None,
        threads: int = 1,
        hash_mb: int = 128,
        stockfish_elo: int = 2000,
    ) -> None:
        if threads <= 0:
            raise ValueError("threads must be positive")

        if hash_mb <= 0:
            raise ValueError("hash_mb must be positive")

        validate_stockfish_elo(stockfish_elo)

        self.model, self.device = load_policy_model(checkpoint_path)

        executable = resolve_stockfish_path(stockfish_path)

        self.engine = chess.engine.SimpleEngine.popen_uci(executable)
        self.engine.configure(
            {
                "Threads": threads,
                "Hash": hash_mb,
                "UCI_LimitStrength": True,
                "UCI_Elo": stockfish_elo,
            }
        )
        log_memory("Stockfish started", logger=LOGGER)

    def choose_move(
        self,
        board: chess.Board,
        *,
        top_k: int = 5,
        mode: SelectionMode = "best_of_top_k",
        temperature: float = 1.0,
        depth: int | None = 12,
        time_limit_seconds: float | None = None,
        style_weight: float | None = None,
    ) -> HybridDecision:
        """Select a move for the current board."""
        proposed = model_candidates(
            self.model,
            board,
            device=self.device,
            top_k=top_k,
            temperature=temperature,
        )

        evaluated = evaluate_candidates(
            self.engine,
            board,
            proposed,
            depth=depth,
            time_limit_seconds=time_limit_seconds,
        )

        resolved_style_weight = phase_style_weight(board) if style_weight is None else style_weight

        if mode == "best_of_top_k":
            selected = choose_best_of_top_k(evaluated)
            final_candidates = evaluated

        elif mode == "blended":
            selected = choose_blended(
                evaluated,
                style_weight=resolved_style_weight,
            )

            final_candidates = evaluated

        else:
            raise ValueError(f"Unsupported selection mode: {mode}")

        return HybridDecision(
            selected_move=selected.move,
            candidates=tuple(final_candidates),
            mode=mode,
            top_k=top_k,
        )

    def close(self) -> None:
        """Shut down the Stockfish process."""
        self.engine.quit()

    def __enter__(self) -> YeafinsHybridEngine:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()
