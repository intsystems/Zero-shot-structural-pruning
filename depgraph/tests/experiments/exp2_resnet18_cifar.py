"""Experiment 2 — SurrogateImportance on ResNet-18 / CIFAR-10.

torchvision ResNet-18 with the CIFAR adaptation (3x3 stride-1 stem, no
maxpool). Skip-connections in each BasicBlock couple the block's output
channels with its input, so DependencyGraph produces one shared "stage"
group per residual stage plus per-block interior groups. The surrogate
graph therefore has real branching and should discriminate stages.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet

from _common import make_common_parser, run_pipeline


def make_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def ignored_layers(model):
    return [m for m in model.modules()
            if isinstance(m, nn.Linear) and m.out_features == 10]


def main():
    args = make_common_parser("exp2: ResNet-18 / CIFAR-10").parse_args()
    run_pipeline(
        experiment_name="exp2_resnet18_cifar10",
        model_fn=make_cifar_resnet18,
        args=args,
        ignored_layers_fn=ignored_layers,
    )


if __name__ == "__main__":
    main()
