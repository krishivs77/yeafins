"""Combine the personalized policy model with Stockfish evaluation."""

from __future__ import annotations

import math
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
from yeafins.training.train import select_device

SelectionMode = Literal["best_of_top_k", "blended"]


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
        weights_only=False,
    )

    model = ResNetPolicy(ResNetPolicyConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

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


def evaluate_candidates(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    candidates: list[tuple[chess.Move, float, int]],
    *,
    depth: int | None = 12,
    time_limit_seconds: float | None = None,
) -> list[CandidateMove]:
    """Evaluate each proposed move with Stockfish."""
    if depth is None and time_limit_seconds is None:
        raise ValueError("Either depth or time_limit_seconds must be provided")

    if depth is not None and depth <= 0:
        raise ValueError("depth must be positive")

    if time_limit_seconds is not None and time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive")

    if time_limit_seconds is not None:
        limit = chess.engine.Limit(
            time=time_limit_seconds,
        )
    else:
        limit = chess.engine.Limit(
            depth=depth,
        )

    evaluated: list[CandidateMove] = []

    for move, probability, rank in candidates:
        if move not in board.legal_moves:
            raise HybridEngineError(f"Candidate move became illegal: {move.uci()}")

        board.push(move)

        try:
            info = engine.analyse(
                board,
                limit,
            )

            child_score = info["score"]
            score_after_move = child_score.pov(not board.turn)

            cp_value = score_after_move.score(
                mate_score=100_000,
            )

            if cp_value is None:
                raise HybridEngineError(f"No numeric score for move {move.uci()}")

        finally:
            board.pop()

        evaluated.append(
            CandidateMove(
                move=move,
                model_probability=probability,
                model_rank=rank,
                stockfish_cp=int(cp_value),
            )
        )

    return evaluated


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
    """Choose using both model preference and Stockfish quality."""
    if not candidates:
        raise HybridEngineError("No candidates were provided")

    if not 0.0 <= style_weight <= 1.0:
        raise ValueError("style_weight must be between 0 and 1")

    engine_weight = 1.0 - style_weight
    engine_scores = normalize_stockfish_scores(candidates)

    rescored: list[CandidateMove] = []

    for candidate in candidates:
        model_score = math.log(max(candidate.model_probability, 1e-12))

        # Convert negative log probabilities to a bounded relative score
        # across this candidate set.
        blended_score = style_weight * model_score + engine_weight * engine_scores[candidate.move]

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
        ),
    )


class YeafinsHybridEngine:
    """Personalized move proposer with Stockfish verification."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        stockfish_path: str | None = None,
        threads: int = 1,
        hash_mb: int = 128,
    ) -> None:
        if threads <= 0:
            raise ValueError("threads must be positive")

        if hash_mb <= 0:
            raise ValueError("hash_mb must be positive")

        self.model, self.device = load_policy_model(checkpoint_path)

        executable = resolve_stockfish_path(stockfish_path)

        self.engine = chess.engine.SimpleEngine.popen_uci(executable)
        self.engine.configure(
            {
                "Threads": threads,
                "Hash": hash_mb,
            }
        )

    def choose_move(
        self,
        board: chess.Board,
        *,
        top_k: int = 5,
        mode: SelectionMode = "best_of_top_k",
        temperature: float = 1.0,
        depth: int | None = 12,
        time_limit_seconds: float | None = None,
        style_weight: float = 0.65,
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

        if mode == "best_of_top_k":
            selected = choose_best_of_top_k(evaluated)
            final_candidates = evaluated

        elif mode == "blended":
            selected = choose_blended(
                evaluated,
                style_weight=style_weight,
            )

            final_candidates = [
                CandidateMove(
                    move=candidate.move,
                    model_probability=candidate.model_probability,
                    model_rank=candidate.model_rank,
                    stockfish_cp=candidate.stockfish_cp,
                    blended_score=(
                        style_weight
                        * math.log(
                            max(
                                candidate.model_probability,
                                1e-12,
                            )
                        )
                        + (1.0 - style_weight)
                        * normalize_stockfish_scores(evaluated)[candidate.move]
                    ),
                )
                for candidate in evaluated
            ]

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
