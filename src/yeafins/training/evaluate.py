"""Evaluate a trained Yeafins policy model on a held-out dataset split."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from yeafins.models.resnet_policy import (
    ResNetPolicy,
    ResNetPolicyConfig,
    apply_legal_move_mask,
)
from yeafins.runtime import select_device
from yeafins.training.dataset import (
    create_dataloader,
    create_legal_mask_tensor,
)


@dataclass
class MetricAccumulator:
    """Accumulate policy-ranking metrics."""

    samples: int = 0
    top1_correct: int = 0
    top3_correct: int = 0
    top5_correct: int = 0
    reciprocal_rank_sum: float = 0.0
    negative_log_likelihood_sum: float = 0.0

    def update(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> None:
        """Update metrics from one batch."""
        batch_size = targets.shape[0]
        self.samples += batch_size

        top5 = logits.topk(
            k=5,
            dim=1,
            largest=True,
            sorted=True,
        ).indices

        matches = top5.eq(targets.unsqueeze(1))

        self.top1_correct += int(matches[:, :1].any(dim=1).sum().item())
        self.top3_correct += int(matches[:, :3].any(dim=1).sum().item())
        self.top5_correct += int(matches[:, :5].any(dim=1).sum().item())

        target_logits = logits.gather(
            dim=1,
            index=targets.unsqueeze(1),
        )

        ranks = 1 + (logits > target_logits).sum(dim=1)
        self.reciprocal_rank_sum += float((1.0 / ranks.float()).sum().item())

        log_probabilities = torch.log_softmax(logits, dim=1)
        target_log_probabilities = log_probabilities.gather(
            dim=1,
            index=targets.unsqueeze(1),
        )

        self.negative_log_likelihood_sum += float((-target_log_probabilities).sum().item())

    def compute(self) -> dict[str, float | int]:
        """Return normalized metrics."""
        if self.samples == 0:
            raise ValueError("Cannot compute metrics with zero samples")

        return {
            "samples": self.samples,
            "top1": self.top1_correct / self.samples,
            "top3": self.top3_correct / self.samples,
            "top5": self.top5_correct / self.samples,
            "mean_reciprocal_rank": (self.reciprocal_rank_sum / self.samples),
            "negative_log_likelihood": (self.negative_log_likelihood_sum / self.samples),
        }


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> ResNetPolicy:
    """Load a trained model from a checkpoint."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model_config = ResNetPolicyConfig(
        **checkpoint["model_config"],
    )

    model = ResNetPolicy(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def rating_band(rating: float) -> str:
    """Group ratings into broad analysis bands."""
    if rating < 600:
        return "under_600"

    if rating < 800:
        return "600_799"

    if rating < 1000:
        return "800_999"

    if rating < 1200:
        return "1000_1199"

    return "1200_plus"


def game_phase(fen: str) -> str:
    """Infer a simple game phase using remaining non-pawn material."""
    import chess

    board = chess.Board(fen)

    material = 0

    piece_values = {
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    for piece_type, value in piece_values.items():
        material += len(board.pieces(piece_type, chess.WHITE)) * value
        material += len(board.pieces(piece_type, chess.BLACK)) * value

    if board.fullmove_number <= 10:
        return "opening"

    if material <= 14:
        return "endgame"

    return "middlegame"


def evaluate(
    checkpoint_path: Path,
    positions_path: Path,
    output_path: Path,
    *,
    split: str = "test",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Evaluate raw and legal-masked policy predictions."""
    device = select_device()
    model = load_model(checkpoint_path, device)

    loader = create_dataloader(
        positions_path=positions_path,
        split=split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    overall_raw = MetricAccumulator()
    overall_masked = MetricAccumulator()

    grouped: dict[str, dict[str, MetricAccumulator]] = {
        "time_class": defaultdict(MetricAccumulator),
        "player_color": defaultdict(MetricAccumulator),
        "rating_band": defaultdict(MetricAccumulator),
        "game_phase": defaultdict(MetricAccumulator),
    }

    prediction_rows: list[dict[str, object]] = []

    with torch.inference_mode():
        for batch in loader:
            boards = batch["board"].to(device)
            targets = batch["target"].to(device)

            raw_logits = model(boards)

            legal_masks = create_legal_mask_tensor(batch["fen"]).to(device)

            masked_logits = apply_legal_move_mask(
                raw_logits,
                legal_masks,
            )

            overall_raw.update(raw_logits, targets)
            overall_masked.update(masked_logits, targets)

            top5_indices = masked_logits.topk(
                k=5,
                dim=1,
            ).indices.cpu()

            targets_cpu = targets.cpu()
            masked_logits_cpu = masked_logits.cpu()

            for index in range(targets.shape[0]):
                sample_logits = masked_logits_cpu[index : index + 1]
                sample_target = targets_cpu[index : index + 1]

                rating = float(batch["player_rating"][index].item())

                group_values = {
                    "time_class": str(batch["time_class"][index]),
                    "player_color": str(batch["player_color"][index]),
                    "rating_band": rating_band(rating),
                    "game_phase": game_phase(str(batch["fen"][index])),
                }

                for group_name, group_value in group_values.items():
                    grouped[group_name][group_value].update(
                        sample_logits,
                        sample_target,
                    )

                prediction_rows.append(
                    {
                        "sample_id": str(batch["sample_id"][index]),
                        "game_id": str(batch["game_id"][index]),
                        "fen": str(batch["fen"][index]),
                        "actual_move": str(batch["move_uci"][index]),
                        "actual_label": int(targets_cpu[index].item()),
                        "predicted_label": int(top5_indices[index, 0].item()),
                        "top5_labels": [int(value) for value in top5_indices[index].tolist()],
                        "actual_rank": int(
                            1
                            + (
                                masked_logits_cpu[index]
                                > masked_logits_cpu[
                                    index,
                                    targets_cpu[index],
                                ]
                            )
                            .sum()
                            .item()
                        ),
                        "time_class": str(batch["time_class"][index]),
                        "player_rating": rating,
                    }
                )

    results: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "device": str(device),
        "raw": overall_raw.compute(),
        "legal_masked": overall_masked.compute(),
        "groups": {},
    }

    for group_name, group_accumulators in grouped.items():
        results["groups"][group_name] = {
            group_value: accumulator.compute()
            for group_value, accumulator in sorted(group_accumulators.items())
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    prediction_path = output_path.with_name(output_path.stem + "_predictions.parquet")

    pd.DataFrame(prediction_rows).to_parquet(
        prediction_path,
        index=False,
    )

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained Yeafins policy model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--positions",
        type=Path,
        default=Path("data/processed/positions.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/resnet_baseline/test_metrics.json"),
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    return parser.parse_args()


def main() -> None:
    """Run model evaluation."""
    args = parse_args()

    results = evaluate(
        checkpoint_path=args.checkpoint,
        positions_path=args.positions,
        output_path=args.output,
        split=args.split,
        batch_size=args.batch_size,
    )

    raw = results["raw"]
    masked = results["legal_masked"]

    print(f"Split:      {results['split']}")
    print(f"Device:     {results['device']}")
    print()
    print("RAW POLICY")
    print(f"Top-1:      {raw['top1']:.4f}")
    print(f"Top-3:      {raw['top3']:.4f}")
    print(f"Top-5:      {raw['top5']:.4f}")
    print(f"MRR:        {raw['mean_reciprocal_rank']:.4f}")
    print(f"NLL:        {raw['negative_log_likelihood']:.4f}")
    print()
    print("LEGAL-MASKED POLICY")
    print(f"Top-1:      {masked['top1']:.4f}")
    print(f"Top-3:      {masked['top3']:.4f}")
    print(f"Top-5:      {masked['top5']:.4f}")
    print(f"MRR:        {masked['mean_reciprocal_rank']:.4f}")
    print(f"NLL:        {masked['negative_log_likelihood']:.4f}")


if __name__ == "__main__":
    main()
