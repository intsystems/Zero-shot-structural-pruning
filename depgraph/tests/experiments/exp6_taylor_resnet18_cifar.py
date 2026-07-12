"""Experiment 6 — TaylorImportance baseline on ResNet-18 / CIFAR-10.

Same architecture and training/finetune schedule as exp2, but the pruner
is driven by `tp.importance.TaylorImportance` instead of the surrogate.
Gradients are accumulated on the calibration subset immediately before
`pruner.step()` — this is handled centrally in `_common.run_pipeline`.

Used as a baseline to compare against the surrogate-based results from
exp2.
"""
from __future__ import annotations

import torch.nn as nn

from _common import make_common_parser, run_pipeline
from exp2_resnet18_cifar import make_cifar_resnet18, ignored_layers

import torch_pruning as tp


def main():
    args = make_common_parser("exp6: Taylor / ResNet-18 / CIFAR-10").parse_args()
    importance = tp.importance.TaylorImportance(normalizer="mean")
    run_pipeline(
        experiment_name="exp6_taylor_resnet18_cifar10",
        model_fn=make_cifar_resnet18,
        args=args,
        ignored_layers_fn=ignored_layers,
        importance=importance,
    )


if __name__ == "__main__":
    main()
