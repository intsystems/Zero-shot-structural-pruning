"""Experiment 7 — TaylorImportance baseline on Small ViT / CIFAR-10.

Same architecture, `num_heads` dict, and training schedule as exp3, but
the pruner is driven by `tp.importance.TaylorImportance`. Gradients are
accumulated on the calibration subset immediately before `pruner.step()`
(handled by `_common.run_pipeline`).

Used as a baseline to compare against the surrogate-based results from
exp3.
"""
from __future__ import annotations

import torch.nn as nn

from _common import make_common_parser, run_pipeline
from exp3_vit_cifar import make_vit, vit_pruner_kwargs

import torch_pruning as tp


def main():
    args = make_common_parser("exp7: Taylor / Small ViT / CIFAR-10").parse_args()
    importance = tp.importance.TaylorImportance(
        normalizer="mean",
        target_types=[
            nn.modules.conv._ConvNd,
            nn.Linear,
            nn.modules.batchnorm._BatchNorm,
            nn.LayerNorm,
        ],
    )
    run_pipeline(
        experiment_name="exp7_taylor_vit_cifar10",
        model_fn=make_vit,
        args=args,
        ignored_layers_fn=lambda m: [m.head],
        pruner_kwargs=vit_pruner_kwargs,
        importance=importance,
    )


if __name__ == "__main__":
    main()
