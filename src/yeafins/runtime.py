"""Lightweight runtime helpers shared by training and inference."""

from __future__ import annotations

import logging
import os
import resource
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

LOGGER = logging.getLogger(__name__)


def select_device() -> torch.device:
    """Select the best locally available PyTorch device.

    The Render image installs CPU-only PyTorch, so Linux production resolves to CPU
    without a deployment-only behavior branch. Local Apple Silicon development keeps
    MPS support, and CUDA remains available in explicitly GPU-enabled environments.
    """
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def memory_logging_enabled() -> bool:
    """Return whether opt-in runtime RSS diagnostics are enabled."""
    return os.getenv("YEAFINS_LOG_MEMORY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def current_rss_mb() -> float:
    """Return current resident memory in MiB without adding a dependency."""
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024

    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return maximum_rss / (1024 * 1024)
    return maximum_rss / 1024


def log_memory(label: str, *, logger: logging.Logger = LOGGER) -> None:
    """Log one labelled RSS checkpoint when diagnostics are enabled."""
    if memory_logging_enabled():
        logger.info("Memory checkpoint: %s rss=%.1f MiB", label, current_rss_mb())
