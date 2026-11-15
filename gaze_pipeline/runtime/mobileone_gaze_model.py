#!/usr/bin/env python3

"""Minimal fused MobileOne-S0 architecture for gaze inference.

This file contains only the MobileOne-S0 inference architecture required
to load the selected 90-bin gaze checkpoint.

Upstream community source:
https://github.com/yakhyo/gaze-estimation
Audited commit:
4bf557d8a819da2b208214a9ef9268fd84cb52f1

Original MobileOne implementation:
https://github.com/apple-aiml-research/ml-mobileone

MobileOne paper:
https://arxiv.org/abs/2206.04040

Modified for this thesis runtime by retaining only the fused MobileOne-S0
inference path and the two gaze-output heads.

Copyright (C) 2022 Apple Inc. All Rights Reserved.
Community modifications copyright (c) 2024 Yakhyokhuja Valikhujaev.

The applicable upstream licence and acknowledgement files must remain
available in the project's third-party notices directory.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


__all__ = [
    "MobileOneBlock",
    "MobileOneGaze",
    "mobileone_s0_gaze",
]


S0_BLOCKS_PER_STAGE = (2, 8, 10, 1)
S0_WIDTH_MULTIPLIERS = (0.75, 1.0, 1.0, 2.0)
DEFAULT_GAZE_BINS = 90


class MobileOneBlock(nn.Module):
    """One fused MobileOne inference block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()

        self.reparam_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True,
        )

        # MobileOne-S0 does not use squeeze-excitation blocks.
        # Keeping the Identity operation matches the audited source.
        self.se = nn.Identity()
        self.activation = nn.ReLU()

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the fused convolution, identity SE and ReLU."""

        outputs = self.reparam_conv(inputs)
        outputs = self.se(outputs)
        return self.activation(outputs)


class MobileOneGaze(nn.Module):
    """Fused MobileOne backbone with yaw and pitch heads."""

    def __init__(
        self,
        num_blocks_per_stage: Sequence[int],
        width_multipliers: Sequence[float],
        num_classes: int = DEFAULT_GAZE_BINS,
    ) -> None:
        super().__init__()

        if len(num_blocks_per_stage) != 4:
            raise ValueError(
                "Expected four MobileOne stage depths."
            )

        if len(width_multipliers) != 4:
            raise ValueError(
                "Expected four MobileOne width multipliers."
            )

        if num_classes < 1:
            raise ValueError(
                "num_classes must be positive."
            )

        self.in_planes = min(
            64,
            int(64 * width_multipliers[0]),
        )

        self.stage0 = MobileOneBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.stage1 = self._make_stage(
            planes=int(
                64 * width_multipliers[0]
            ),
            num_blocks=int(
                num_blocks_per_stage[0]
            ),
        )

        self.stage2 = self._make_stage(
            planes=int(
                128 * width_multipliers[1]
            ),
            num_blocks=int(
                num_blocks_per_stage[1]
            ),
        )

        self.stage3 = self._make_stage(
            planes=int(
                256 * width_multipliers[2]
            ),
            num_blocks=int(
                num_blocks_per_stage[2]
            ),
        )

        self.stage4 = self._make_stage(
            planes=int(
                512 * width_multipliers[3]
            ),
            num_blocks=int(
                num_blocks_per_stage[3]
            ),
        )

        self.gap = nn.AdaptiveAvgPool2d(
            output_size=1
        )

        final_channels = int(
            512 * width_multipliers[3]
        )

        self.fc_yaw = nn.Linear(
            final_channels,
            num_classes,
        )

        self.fc_pitch = nn.Linear(
            final_channels,
            num_classes,
        )

    def _make_stage(
        self,
        planes: int,
        num_blocks: int,
    ) -> nn.Sequential:
        """Build one fused MobileOne stage."""

        if num_blocks < 1:
            raise ValueError(
                "A stage must contain at least one block."
            )

        strides = [2] + [1] * (
            num_blocks - 1
        )
        blocks: list[nn.Module] = []

        for stride in strides:
            # Depthwise convolution.
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=self.in_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=self.in_planes,
                )
            )

            # Pointwise convolution.
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                )
            )

            self.in_planes = planes

        return nn.Sequential(*blocks)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return yaw logits first and pitch logits second."""

        outputs = self.stage0(inputs)
        outputs = self.stage1(outputs)
        outputs = self.stage2(outputs)
        outputs = self.stage3(outputs)
        outputs = self.stage4(outputs)
        outputs = self.gap(outputs)
        outputs = outputs.reshape(
            outputs.size(0),
            -1,
        )

        yaw_logits = self.fc_yaw(outputs)
        pitch_logits = self.fc_pitch(outputs)

        return yaw_logits, pitch_logits


def mobileone_s0_gaze(
    num_classes: int = DEFAULT_GAZE_BINS,
) -> MobileOneGaze:
    """Construct the fused MobileOne-S0 gaze architecture."""

    return MobileOneGaze(
        num_blocks_per_stage=(
            S0_BLOCKS_PER_STAGE
        ),
        width_multipliers=(
            S0_WIDTH_MULTIPLIERS
        ),
        num_classes=num_classes,
    )
