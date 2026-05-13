import torch
from torch import nn
from torch.fx import symbolic_trace
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader, Subset, random_split
import torchvision
from torchvision import transforms
import torch.fx as fx
from torchvision.models import resnet50

import numpy as np
from scipy.stats import spearmanr, kendalltau
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import abc
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from functools import reduce, partial
import re
import copy
import collections
from collections import defaultdict
from __future__ import annotations
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm.notebook import tqdm, trange
import networkx as nx

sns.set_style('darkgrid')

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
