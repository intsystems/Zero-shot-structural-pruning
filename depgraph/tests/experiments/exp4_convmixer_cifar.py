"""Experiment 4 — SurrogateImportance on ConvMixer / CIFAR-10.

Transformer-adjacent block: patch embedding, then repeated
`(depthwise conv + pointwise conv)` blocks with LN/GELU/BN. No attention,
so groups stay Conv-rooted. Depthwise convs make each channel its own
input-group, so `MetaPruner` treats out-channels as coupled with
downstream pointwise-conv in-channels — the resulting group graph has
many independent Conv-rooted groups per block.
"""
from __future__ import annotations

import torch.nn as nn

from _common import make_common_parser, run_pipeline


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


def make_convmixer(dim: int = 128, depth: int = 8, kernel: int = 5,
                   patch: int = 4, num_classes: int = 10) -> nn.Module:
    layers = [
        nn.Conv2d(3, dim, kernel_size=patch, stride=patch),
        nn.GELU(),
        nn.BatchNorm2d(dim),
    ]
    for _ in range(depth):
        layers.append(Residual(nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel, groups=dim, padding=kernel // 2),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )))
        layers.append(nn.Conv2d(dim, dim, kernel_size=1))
        layers.append(nn.GELU())
        layers.append(nn.BatchNorm2d(dim))
    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(dim, num_classes))
    return nn.Sequential(*layers)


def ignored_layers(model):
    return [m for m in model.modules()
            if isinstance(m, nn.Linear) and m.out_features == 10]


def main():
    args = make_common_parser("exp4: ConvMixer / CIFAR-10").parse_args()
    run_pipeline(
        experiment_name="exp4_convmixer_cifar10",
        model_fn=lambda: make_convmixer(dim=128, depth=8),
        args=args,
        ignored_layers_fn=ignored_layers,
    )


if __name__ == "__main__":
    main()
