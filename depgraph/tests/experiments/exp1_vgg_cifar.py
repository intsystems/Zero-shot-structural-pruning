"""Experiment 1 — SurrogateImportance on VGG-11 / CIFAR-10.

Plain sequential CNN with BatchNorm, no residual connections. This is the
low-branching case: the DependencyGraph forms a chain of groups, and the
group-level surrogate graph has a single outgoing edge per source vertex,
so softmax(gamma) is degenerate (all 1.0). Expected outcome: raw surrogate
scores are near-identical across groups, and pruning is driven mostly by
the per-channel L1 tie-breaker inside `SurrogateImportance.__call__`.
"""
from __future__ import annotations

import torch.nn as nn

from _common import make_common_parser, run_pipeline


def make_cifar_vgg11(num_classes: int = 10) -> nn.Module:
    cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))
            in_channels = v
    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(512, num_classes))
    return nn.Sequential(*layers)


def ignored_layers(model):
    return [m for m in model.modules()
            if isinstance(m, nn.Linear) and m.out_features == 10]


def main():
    args = make_common_parser("exp1: VGG-11 / CIFAR-10").parse_args()
    run_pipeline(
        experiment_name="exp1_vgg11_cifar10",
        model_fn=make_cifar_vgg11,
        args=args,
        ignored_layers_fn=ignored_layers,
    )


if __name__ == "__main__":
    main()
