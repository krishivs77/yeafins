"""Checkpoint utilities shared by inference and maintenance scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

INFERENCE_FORMAT_VERSION = 1


def export_inference_checkpoint(source: Path, destination: Path) -> None:
    """Copy only model configuration and weights from a training checkpoint."""
    checkpoint: dict[str, Any] = torch.load(
        source,
        map_location="cpu",
        weights_only=True,
    )
    required = {"model_config", "model_state_dict"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {sorted(missing)}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": INFERENCE_FORMAT_VERSION,
            "model_config": checkpoint["model_config"],
            "model_state_dict": checkpoint["model_state_dict"],
        },
        destination,
    )
