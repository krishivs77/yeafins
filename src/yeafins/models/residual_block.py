"""Residual building blocks for the Yeafins chess policy network."""

from __future__ import annotations

from torch import Tensor, nn


class ResidualBlock(nn.Module):
    """A two-convolution residual block preserving spatial dimensions."""

    def __init__(
        self,
        channels: int,
        *,
        batch_norm_momentum: float = 0.1,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive")

        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(
            channels,
            momentum=batch_norm_momentum,
        )
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(
            channels,
            momentum=batch_norm_momentum,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the residual transformation."""
        residual = inputs

        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)

        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)

        outputs = outputs + residual
        outputs = self.relu(outputs)

        return outputs
