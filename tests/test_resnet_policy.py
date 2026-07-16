"""Tests for the residual chess policy network."""

import pytest
import torch
from torch import nn

from yeafins.data.board import BOARD_PLANES
from yeafins.data.encode import POLICY_SIZE
from yeafins.models.residual_block import ResidualBlock
from yeafins.models.resnet_policy import (
    ResNetPolicy,
    ResNetPolicyConfig,
    apply_legal_move_mask,
    count_trainable_parameters,
    legal_move_probabilities,
)


def make_small_model() -> ResNetPolicy:
    """Create a small model suitable for fast unit tests."""
    return ResNetPolicy(
        ResNetPolicyConfig(
            trunk_channels=16,
            residual_blocks=2,
        )
    )


def test_residual_block_preserves_shape() -> None:
    block = ResidualBlock(channels=16)
    inputs = torch.randn(4, 16, 8, 8)

    outputs = block(inputs)

    assert outputs.shape == inputs.shape


def test_resnet_policy_forward_shape() -> None:
    model = make_small_model()
    boards = torch.randn(4, BOARD_PLANES, 8, 8)

    logits = model(boards)

    assert logits.shape == (4, POLICY_SIZE)
    assert logits.dtype == torch.float32


def test_default_model_has_expected_configuration() -> None:
    model = ResNetPolicy()

    assert model.config.input_channels == BOARD_PLANES
    assert model.config.trunk_channels == 64
    assert model.config.residual_blocks == 6
    assert model.config.policy_channels == 73


def test_model_has_trainable_parameters() -> None:
    model = make_small_model()

    parameter_count = count_trainable_parameters(model)

    assert parameter_count > 0
    assert parameter_count == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def test_forward_supports_backpropagation() -> None:
    torch.manual_seed(42)

    model = make_small_model()
    boards = torch.randn(8, BOARD_PLANES, 8, 8)
    targets = torch.randint(
        low=0,
        high=POLICY_SIZE,
        size=(8,),
    )

    logits = model(boards)
    loss = nn.CrossEntropyLoss()(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]

    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_forward_rejects_incorrect_board_shape() -> None:
    model = make_small_model()
    boards = torch.randn(4, BOARD_PLANES, 7, 8)

    with pytest.raises(ValueError, match="Expected board dimensions"):
        model(boards)


def test_forward_rejects_integer_input() -> None:
    model = make_small_model()
    boards = torch.zeros(
        4,
        BOARD_PLANES,
        8,
        8,
        dtype=torch.long,
    )

    with pytest.raises(TypeError, match="floating-point"):
        model(boards)


def test_legal_move_mask_sets_illegal_logits_to_negative_infinity() -> None:
    logits = torch.zeros(2, POLICY_SIZE)
    legal_masks = torch.zeros(
        2,
        POLICY_SIZE,
        dtype=torch.bool,
    )

    legal_masks[0, 10] = True
    legal_masks[0, 20] = True
    legal_masks[1, 30] = True

    masked = apply_legal_move_mask(logits, legal_masks)

    assert torch.isfinite(masked[0, 10])
    assert torch.isfinite(masked[0, 20])
    assert torch.isfinite(masked[1, 30])

    assert torch.isneginf(masked[0, 0])
    assert torch.isneginf(masked[1, 0])


def test_legal_move_probabilities_sum_to_one() -> None:
    logits = torch.randn(2, POLICY_SIZE)
    legal_masks = torch.zeros(
        2,
        POLICY_SIZE,
        dtype=torch.bool,
    )

    legal_masks[0, [10, 20, 30]] = True
    legal_masks[1, [100, 200]] = True

    probabilities = legal_move_probabilities(
        logits,
        legal_masks,
    )

    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones(2),
    )

    assert probabilities[0, 0] == 0.0
    assert probabilities[1, 0] == 0.0


def test_mask_rejects_position_without_legal_moves() -> None:
    logits = torch.zeros(1, POLICY_SIZE)
    legal_masks = torch.zeros(
        1,
        POLICY_SIZE,
        dtype=torch.bool,
    )

    with pytest.raises(ValueError, match="at least one legal move"):
        apply_legal_move_mask(logits, legal_masks)


def test_invalid_model_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        ResNetPolicy(
            ResNetPolicyConfig(
                trunk_channels=0,
            )
        )
