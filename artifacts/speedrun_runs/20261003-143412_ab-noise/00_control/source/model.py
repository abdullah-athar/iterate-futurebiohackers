"""ResNet9-style network for 32x32 images (the DAWNBench CIFAR-10 design, 100 classes).

Plain PyTorch: 3x3 convolutions without bias, BatchNorm, ReLU, max pooling, two
residual blocks, a global max pool and one linear layer. No pretrained weights.
Input normalization lives inside forward() so the evaluator's [0, 1] float images
and the training batches follow the same convention.
"""

from __future__ import annotations

import torch
from torch import nn

# Logit scale from the DAWNBench ResNet9 recipe. A fixed scalar, not a learned tensor.
OUTPUT_SCALE = 0.125


def conv_bn(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class Residual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(conv_bn(channels, channels), conv_bn(channels, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNet9(nn.Module):
    """Channels (w, 2w, 4w, 8w); width 64 gives the classic 64-128-256-512 network."""

    def __init__(self, num_classes: int = 100, width: int = 64, channels_last: bool = False):
        super().__init__()
        w = width
        self.channels_last = channels_last
        # Per-channel input statistics. Identity until prepare() fills them from the
        # training split; recomputed for every trial, never learned.
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("std", torch.ones(1, 3, 1, 1))
        self.features = nn.Sequential(
            conv_bn(3, w),  # 32x32
            conv_bn(w, 2 * w),
            nn.MaxPool2d(2),  # 16x16
            Residual(2 * w),
            conv_bn(2 * w, 4 * w),
            nn.MaxPool2d(2),  # 8x8
            conv_bn(4 * w, 8 * w),
            nn.MaxPool2d(2),  # 4x4
            Residual(8 * w),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(8 * w, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, 32, 32] float in [0, 1] (any B >= 1). Returns logits [B, num_classes]."""
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x = (x - self.mean) / self.std
        return self.classifier(self.features(x)) * OUTPUT_SCALE
