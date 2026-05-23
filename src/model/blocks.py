from __future__ import annotations

import torch
from torch import nn


class LinearBlock(nn.Module):
    """
    Node for linear with activation.
    """

    def __init__(self, in_features, out_features, activation=nn.ReLU, is_end=True):
        super().__init__()
        self.out_features = out_features
        self.is_end = is_end
        self.linear = nn.Linear(in_features, out_features)
        self.act = activation()

    def forward(self, x):
        return self.act(self.linear(x))

class ConvBlock(nn.Module):
    """
    Node for convolution with activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=nn.ReLU, is_end=True):
        super().__init__()
        self.out_features = out_channels
        self.is_end = is_end

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.act = activation()

    def forward(self, x):
        return self.act(self.conv(x))

class FlattenBlock(nn.Module):
    def __init__(self, in_channels, spatial, is_end=False):
        super().__init__()
        self.out_features = in_channels * spatial * spatial
        self.is_end = is_end

    def forward(self, x):
        return torch.flatten(x, 1)

class IdBlock(nn.Module):
    """
    Node for identical function. Used for input.
    """

    def __init__(self, input_dim, is_end=False):
        super().__init__()
        self.is_end = is_end
        self.out_features = input_dim

    def forward(self, x):
        # return x
        return torch.flatten(x, 1)
