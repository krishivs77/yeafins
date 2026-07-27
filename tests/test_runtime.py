"""Tests for inference-safe runtime helpers and checkpoint loading."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import chess
import pytest
import torch

from yeafins.checkpoints import export_inference_checkpoint
from yeafins.engine.hybrid import load_policy_model, model_candidates
from yeafins.models.resnet_policy import ResNetPolicy, ResNetPolicyConfig
from yeafins.runtime import current_rss_mb, log_memory, select_device

ROOT = Path(__file__).parents[1]


def make_training_checkpoint(path: Path) -> ResNetPolicy:
    model = ResNetPolicy(ResNetPolicyConfig(trunk_channels=8, residual_blocks=1))
    torch.save(
        {
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {"state": {"unneeded": torch.ones(64)}},
            "scheduler_state_dict": {"unneeded": True},
            "history": [{"epoch": 1}],
        },
        path,
    )
    return model


def test_api_import_does_not_load_training_or_data_dependencies() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; import yeafins.api.app; "
            "blocked=('pandas','pyarrow','sklearn','yeafins.training'); "
            "loaded=[name for name in sys.modules if name.startswith(blocked)]; "
            "assert not loaded, loaded"
        ),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def test_select_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device() == torch.device("cpu")


def test_export_and_load_inference_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "training.pt"
    destination = tmp_path / "inference.pt"
    original = make_training_checkpoint(source)

    export_inference_checkpoint(source, destination)
    exported = torch.load(destination, map_location="cpu", weights_only=True)

    assert set(exported) == {"format_version", "model_config", "model_state_dict"}
    assert "optimizer_state_dict" not in exported
    assert destination.stat().st_size < source.stat().st_size

    loaded, device = load_policy_model(destination)
    assert device == torch.device("cpu")
    assert loaded.training is False
    assert not hasattr(loaded, "optimizer")

    board = chess.Board()
    original.eval()
    expected = model_candidates(
        original,
        board,
        device=torch.device("cpu"),
        top_k=5,
    )
    actual = model_candidates(loaded, board, device=device, top_k=5)
    assert [(move, rank) for move, _, rank in actual] == [
        (move, rank) for move, _, rank in expected
    ]


def test_memory_logging_is_optional(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test-memory")
    monkeypatch.delenv("YEAFINS_LOG_MEMORY", raising=False)
    log_memory("hidden", logger=logger)
    assert "hidden" not in caplog.text

    monkeypatch.setenv("YEAFINS_LOG_MEMORY", "true")
    with caplog.at_level(logging.INFO, logger="test-memory"):
        log_memory("visible", logger=logger)
    assert "Memory checkpoint: visible rss=" in caplog.text
    assert current_rss_mb() > 0


def test_cpu_runtime_verifier_has_no_false_positives() -> None:
    environment = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "scripts/verify_cpu_runtime.py"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "Unexpected GPU dependencies installed" in result.stderr:
        pytest.skip("Local development environment intentionally contains GPU packages")
    assert result.returncode == 0, result.stderr
