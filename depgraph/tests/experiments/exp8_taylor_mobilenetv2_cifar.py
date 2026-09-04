"""Experiment 8 — TaylorImportance baseline on MobileNetV2 / CIFAR-10.

Same training/finetune schedule as exp2/exp6, but the model is a CIFAR-10
adapted torchvision MobileNetV2 (3x3 stride-1 stem). InvertedResidual blocks
mix residual and concat coupling, so the DependencyGraph has to merge
skip-concat groups — a good stress test for a plain magnitude-style
criterion. The pruner is driven by `tp.importance.TaylorImportance`;
gradients are accumulated on the calibration subset immediately before
`pruner.step()` — this is handled centrally in `_common.run_pipeline`.

Used as a baseline to compare against the surrogate-based results.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision.models.mobilenet_v2 import MobileNetV2

from _common import make_common_parser, run_pipeline

import torch_pruning as tp


def make_cifar_mobilenetv2(num_classes: int = 10) -> nn.Module:
    model = MobileNetV2(num_classes=num_classes)
    # CIFAR adaptation: 3x3 stride-1 stem instead of the 7x7 stride-2 one.
    # MobileNetV2 has no max-pool — downsampling is done by stride-2 blocks,
    # which are kept as is (32x32 -> 2x2 before the classifier).
    model.features[0][0] = nn.Conv2d(
        3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    return model


def ignored_layers(model):
    return [m for m in model.modules()
            if isinstance(m, nn.Linear) and m.out_features == 10]


def main():
    args = make_common_parser("exp8: Taylor / MobileNetV2 / CIFAR-10").parse_args()
    importance = tp.importance.TaylorImportance(normalizer="mean")
    run_pipeline(
        experiment_name="exp8_taylor_mobilenetv2_cifar10",
        model_fn=make_cifar_mobilenetv2,
        args=args,
        ignored_layers_fn=ignored_layers,
        importance=importance,
    )


if __name__ == "__main__":
    main()
