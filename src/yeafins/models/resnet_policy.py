"""Residual convolutional policy model for personalized chess move prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from yeafins.data.board import BOARD_PLANES
from yeafins.data.encode import POLICY_PLANES, POLICY_SIZE
from yeafins.models.residual_block import ResidualBlock


@dataclass(frozen=True)
class ResNetPolicyConfig:
    """Configuration for a Yeafins residual policy network."""

    input_channels: int = BOARD_PLANES
    trunk_channels: int = 64
    residual_blocks: int = 6
    policy_channels: int = POLICY_PLANES
    batch_norm_momentum: float = 0.1

    def validate(self) -> None:
        """Validate architecture dimensions."""
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")

        if self.trunk_channels <= 0:
            raise ValueError("trunk_channels must be positive")

        if self.residual_blocks <= 0:
            raise ValueError("residual_blocks must be positive")

        if self.policy_channels != POLICY_PLANES:
            raise ValueError(f"policy_channels must equal {POLICY_PLANES}")

        if not 0.0 < self.batch_norm_momentum <= 1.0:
            raise ValueError("batch_norm_momentum must be in the interval (0, 1]")

    def to_dict(self) -> dict[str, int | float]:
        """Return a serializable model configuration."""
        return asdict(self)


class ResNetPolicy(nn.Module):
    """Predict move logits over the fixed 4,672-class policy space."""

    def __init__(
        self,
        config: ResNetPolicyConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config or ResNetPolicyConfig()
        self.config.validate()

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=self.config.input_channels,
                out_channels=self.config.trunk_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                self.config.trunk_channels,
                momentum=self.config.batch_norm_momentum,
            ),
            nn.ReLU(inplace=True),
        )

        self.residual_tower = nn.Sequential(
            *[
                ResidualBlock(
                    self.config.trunk_channels,
                    batch_norm_momentum=(self.config.batch_norm_momentum),
                )
                for _ in range(self.config.residual_blocks)
            ]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=self.config.trunk_channels,
                out_channels=self.config.trunk_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(
                self.config.trunk_channels,
                momentum=self.config.batch_norm_momentum,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.config.trunk_channels,
                out_channels=self.config.policy_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize convolution and normalization parameters."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, boards: Tensor) -> Tensor:
        """Return raw policy logits with shape [batch, 4672]."""
        self._validate_input(boards)

        features = self.stem(boards)
        features = self.residual_tower(features)
        policy_planes = self.policy_head(features)

        # Conv2d returns:
        # [batch, policy_plane, rank, file]
        #
        # Our policy encoding expects:
        # [batch, rank, file, policy_plane]
        #
        # Flattening rank then file corresponds to python-chess square
        # indices a1=0, b1=1, ..., h8=63.
        logits = policy_planes.permute(0, 2, 3, 1).contiguous().view(boards.shape[0], POLICY_SIZE)

        return logits

    def _validate_input(self, boards: Tensor) -> None:
        """Validate a board tensor before inference."""
        if boards.ndim != 4:
            raise ValueError("boards must have shape [batch, channels, 8, 8]")

        expected_shape = (
            self.config.input_channels,
            8,
            8,
        )

        if tuple(boards.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected board dimensions {expected_shape}, received {tuple(boards.shape[1:])}"
            )

        if not boards.is_floating_point():
            raise TypeError("boards must use a floating-point dtype")


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a PyTorch module."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def apply_legal_move_mask(
    logits: Tensor,
    legal_masks: Tensor,
) -> Tensor:
    """Mask illegal policy classes before move selection.

    Args:
        logits:
            Raw model outputs shaped [batch, 4672].
        legal_masks:
            Boolean tensors of the same shape. True indicates a legal move.

    Returns:
        A new tensor where illegal logits are negative infinity.
    """
    if logits.ndim != 2 or logits.shape[1] != POLICY_SIZE:
        raise ValueError(f"logits must have shape [batch, {POLICY_SIZE}]")

    if legal_masks.shape != logits.shape:
        raise ValueError("legal_masks must have the same shape as logits")

    if legal_masks.dtype != torch.bool:
        raise TypeError("legal_masks must have dtype torch.bool")

    if not legal_masks.any(dim=1).all():
        raise ValueError("Every position must contain at least one legal move")

    return logits.masked_fill(~legal_masks, float("-inf"))


def legal_move_probabilities(
    logits: Tensor,
    legal_masks: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Convert raw logits into probabilities over legal moves only."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    masked_logits = apply_legal_move_mask(
        logits / temperature,
        legal_masks,
    )

    return torch.softmax(masked_logits, dim=1)
