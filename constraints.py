"""Utility layers for bounded weights and activation scaling."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def scale_to_unit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scale tensor to [-1, 1] by dividing by its max absolute value.

    Args:
        x: Input tensor.
        eps: Small constant for numerical stability.
    """
    max_abs = x.abs().amax(dim=tuple(range(1, x.dim())), keepdim=True)
    return x / (max_abs + eps)


class ConstrainedLinear(nn.Module):
    """Linear layer with tanh-reparameterized weights in [-scale, scale]."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight_param = nn.Parameter(torch.empty(out_features, in_features))
        self.scale = float(scale)
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias_param", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight_param)
        if self.bias_param is not None:
            nn.init.zeros_(self.bias_param)

    @property
    def weight(self) -> torch.Tensor:
        return torch.tanh(self.weight_param) * self.scale

    @property
    def bias(self) -> Optional[torch.Tensor]:
        if self.bias_param is None:
            return None
        return torch.tanh(self.bias_param) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.nn.functional.linear(x, self.weight, self.bias)


class ConstrainedConv1d(nn.Module):
    """Conv1d with tanh-reparameterized weights."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight_param = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.scale = float(scale)
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias_param", None)
        self.stride = stride
        self.padding = padding
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_param, a=5 ** 0.5)
        if self.bias_param is not None:
            nn.init.zeros_(self.bias_param)

    @property
    def weight(self) -> torch.Tensor:
        return torch.tanh(self.weight_param) * self.scale

    @property
    def bias(self) -> Optional[torch.Tensor]:
        if self.bias_param is None:
            return None
        return torch.tanh(self.bias_param) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.nn.functional.conv1d(
            x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding
        )
