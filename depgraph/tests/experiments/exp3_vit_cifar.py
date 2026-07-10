"""Experiment 3 — SurrogateImportance on a small ViT / CIFAR-10.

Custom small Vision Transformer with fused-QKV attention so `MetaPruner`
can detect attention groups via its `num_heads` dict. Two things are
pruned structurally: (a) MLP hidden-dim inside each block, (b) head_dim
inside each attention head (num_heads kept constant — flip
`prune_num_heads=True` in the closure below if you also want to drop
entire heads).

The surrogate scores Linear/LayerNorm/Conv-rooted groups. LayerNorm is
added on top of the default target_types because most ViT groups are
LN- or Linear-rooted.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from _common import make_common_parser, run_pipeline

import torch_pruning as tp


class SimpleAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, H, N, D)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SimpleAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SmallViT(nn.Module):
    def __init__(self, img_size: int = 32, patch: int = 4,
                 dim: int = 192, depth: int = 6, num_heads: int = 6,
                 num_classes: int = 10):
        super().__init__()
        assert img_size % patch == 0
        n_patches = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.blocks = nn.ModuleList([ViTBlock(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def make_vit() -> nn.Module:
    return SmallViT(img_size=32, patch=4, dim=192, depth=6, num_heads=6, num_classes=10)


def vit_pruner_kwargs(model):
    num_heads = {m.qkv: m.num_heads for m in model.modules()
                 if isinstance(m, SimpleAttention)}
    return dict(
        num_heads=num_heads,
        prune_num_heads=False,
        prune_head_dims=True,
        unwrapped_parameters=[(model.cls_token, 2), (model.pos_embed, 2)],
    )


def main():
    args = make_common_parser("exp3: Small ViT / CIFAR-10").parse_args()
    importance = tp.importance.SurrogateImportance(
        surrogate_epochs=args.surrogate_epochs,
        surrogate_lr=args.surrogate_lr,
        surrogate_batch_size=32,
        normalizer="mean",
        target_types=(
            nn.modules.conv._ConvNd,
            nn.Linear,
            nn.modules.batchnorm._BatchNorm,
            nn.LayerNorm,
        ),
    )
    run_pipeline(
        experiment_name="exp3_vit_cifar10",
        model_fn=make_vit,
        args=args,
        ignored_layers_fn=lambda m: [m.head],
        pruner_kwargs=vit_pruner_kwargs,
        importance=importance,
    )


if __name__ == "__main__":
    main()
