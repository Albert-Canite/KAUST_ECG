"""Segment-aware student model with photonic MLP head."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from constraints import ConstrainedConv1d, ConstrainedLinear, scale_to_unit


class SegmentAwareStudent(nn.Module):
    """Student network with segment-specific Conv1d encoders and lightweight MLP."""

    def __init__(
        self,
        num_classes: int = 2,
        num_mlp_layers: int = 1,
        dropout_rate: float = 0.1,
        use_value_constraint: bool = True,
        use_tanh_activations: bool = False,
        constraint_scale: float = 1.0,
        use_bias: bool = True,
        use_constrained_classifier: bool = False,
    ) -> None:
        super().__init__()
        mlp_layers = max(1, num_mlp_layers)
        self.use_value_constraint = use_value_constraint
        self.use_tanh_activations = use_tanh_activations

        conv_layer = ConstrainedConv1d if use_value_constraint else nn.Conv1d
        conv_kwargs = dict(kernel_size=4, stride=1, padding=0, bias=use_bias)
        if use_value_constraint:
            conv_kwargs["scale"] = constraint_scale

        self.conv_p = conv_layer(1, 4, **conv_kwargs)
        self.conv_qrs = conv_layer(1, 4, **conv_kwargs)
        self.conv_t = conv_layer(1, 4, **conv_kwargs)
        self.conv_global = conv_layer(1, 4, **conv_kwargs)

        linear_cls = ConstrainedLinear if use_value_constraint else nn.Linear
        self.mlp_layers = nn.ModuleList()
        for _ in range(mlp_layers):
            layer_kwargs = {"bias": use_bias, "scale": constraint_scale} if use_value_constraint else {"bias": use_bias}
            self.mlp_layers.append(linear_cls(4, 4, **layer_kwargs))

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.token_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        if use_constrained_classifier:
            classifier_kwargs = {"bias": use_bias, "scale": constraint_scale} if use_value_constraint else {"bias": use_bias}
            self.classifier = ConstrainedLinear(4, num_classes, **classifier_kwargs)
        else:
            self.classifier = nn.Linear(4, num_classes, bias=use_bias)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x) if self.use_tanh_activations else F.relu(x)

    def _pool_tokens(self, x: torch.Tensor, num_blocks: int) -> torch.Tensor:
        pool = nn.AvgPool1d(kernel_size=x.shape[-1] // num_blocks, stride=x.shape[-1] // num_blocks)
        pooled = pool(x)
        return pooled.transpose(1, 2)

    def _process_segment(self, x: torch.Tensor, conv: nn.Module, num_blocks: int) -> torch.Tensor:
        out = self._activate(conv(x))
        tokens = self._pool_tokens(out, num_blocks)
        return tokens

    def _scale_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_value_constraint or self.use_tanh_activations:
            return x
        return scale_to_unit(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p_seg = x[:, :, 0:120]
        qrs_seg = x[:, :, 120:240]
        t_seg = x[:, :, 240:360]
        global_seg = x

        tokens: List[torch.Tensor] = []
        tokens.append(self._process_segment(p_seg, self.conv_p, num_blocks=2))
        tokens.append(self._process_segment(qrs_seg, self.conv_qrs, num_blocks=3))
        tokens.append(self._process_segment(t_seg, self.conv_t, num_blocks=2))

        g_out = self._activate(self.conv_global(global_seg))
        g_token = g_out.mean(dim=-1, keepdim=True).transpose(1, 2)
        tokens.append(g_token)

        h0 = torch.cat(tokens, dim=1)  # (batch, 8, 4)

        h = h0
        for layer in self.mlp_layers:
            h = self._scale_if_needed(h)
            h = layer(h)
            h = self._activate(h)
            h = self.token_dropout(h)

        h_pool = h.mean(dim=1)
        h_pool = self.dropout(h_pool)
        logits = self.classifier(h_pool)
        return logits, h_pool
