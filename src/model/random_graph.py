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

from model.blocks import LinearBlock, ConvBlock, FlattenBlock, IdBlock
from utils import update_log

class RandomGraphModel(nn.Module):
    def __init__(self, modules, edges, collect_stats=True):
        super().__init__()
        self.layers = nn.ModuleList(modules)
        self.edges = edges  # list (i, j)
        self.collect_stats = collect_stats

        if collect_stats:
            self.register_buffer(
                "edge_l1_norms",
                torch.zeros(len(edges))
            )

            self.edge_gates = nn.Parameter(
                torch.ones(len(edges)),
                requires_grad=True
            )

            self.register_buffer(
                "edge_molchanov_importance",
                torch.zeros(len(edges))
            )
    
    def reset_edge_stats(self):
        if self.collect_stats:
            self.edge_l1_norms.zero_()
            self.edge_molchanov_importance.zero_()
    
    def accumulate_importance(self):
        """
        After backward()
        """
        if not self.collect_stats:
            return
        
        with torch.no_grad():
            if self.edge_gates.grad is None:
                return

            g = self.edge_gates.grad

            # Taylor FO importance
            importance = (g * self.edge_gates) ** 2

            self.edge_molchanov_importance += importance

            # Delete gradients
            self.edge_gates.grad.zero_()

    def forward(self, x, edge_mask):
        """
        x: input tensor
        edge_mask: Tensor [num_edges]
        """
        node_outputs = {}

        node_outputs[-1] = x

        for k, (i, j) in enumerate(self.edges):
            if i not in node_outputs:
                continue
            
            out = self.layers[j](node_outputs[i])

            if self.collect_stats:
                # Calculating norm
                norm = out.detach().abs().mean()
                self.edge_l1_norms[k] += norm

                # Applying gate for Molchanov
                out = out * self.edge_gates[k]

            out = out * edge_mask[k]

            if j in node_outputs:
                node_outputs[j] = node_outputs[j] + out
            else:
                node_outputs[j] = out

        return node_outputs[max(node_outputs.keys())]


def generate_graph(input_dim, n_iterations):
    """
    Generate NN for MNIST classification. 
    Returns lists with modules and edges between them.
    """

    # Init lists
    modules = [IdBlock(input_dim), LinearBlock(input_dim, 400)]
    edges = [(-1, 0), (0, 1)]

    # Iteration loop of generation
    tbar = trange(n_iterations, desc='Generating graph', leave=False)
    for i in tbar:
        update_log(i, 'generate_graph')
        
        # Choosing parent modeules
        parent1, parent2 = np.random.randint(len(modules), size=2)

        # Random dimention for new modules
        out_dim_new_block = 2 ** np.random.randint(3, 9)

        # Creating the first module
        out_dim_1 = modules[parent1].out_features
        new_block_1 = LinearBlock(out_dim_1, out_dim_new_block, is_end=False)
        edges.append((parent1, len(modules)))
        modules.append(new_block_1)
        modules[parent1].is_end = False

        # Creating the second module
        out_dim_2 = modules[parent2].out_features
        new_block_2 = LinearBlock(out_dim_2, out_dim_new_block, is_end=False)
        edges.append((parent2, len(modules)))
        modules.append(new_block_2)
        modules[parent2].is_end = False

        # Choosing and creating aggregating module
        out_dim_sum_block = 2 ** np.random.randint(3, 9)
        sum_block = LinearBlock(out_dim_new_block, out_dim_sum_block)
        edges.extend([
            (len(modules)-1, len(modules)),
            (len(modules)-2, len(modules))
            ])
        modules.append(sum_block)
    
    # After generation, there are a lot of dangling vertices - let's fix this.

    # Reduce such vertices to a single dimension for subsequent aggregation
    end_idxs = []
    for i, module in enumerate(modules):
        if module.is_end:
            end_block = LinearBlock(module.out_features, 20, is_end=False)
            end_idxs.append(len(modules))
            edges.append((i, len(modules)))
            modules.append(end_block)
            module.is_end = False
    
    # Creating the final classifier
    classifier = nn.Linear(20, 10)
    edges.extend([
        (end_idx, len(modules)) for end_idx in end_idxs
        ])
    modules.append(classifier)

    return modules, edges


def generate_conv_graph(n_iterations):

    modules = [IdBlock(1), ConvBlock(1, 32)]

    edges = [(-1, 0), (0, 1)]

    for i in trange(n_iterations, desc="Generating conv graph", leave=False):
        update_log(i, 'generate_conv_graph')

        parent1, parent2 = np.random.randint(len(modules), size=2)

        out_channels = 2 ** np.random.randint(3, 6)

        # first conv
        in_ch = modules[parent1].out_features
        block1 = ConvBlock(in_ch, out_channels, is_end=False)

        edges.append((parent1, len(modules)))
        modules.append(block1)
        modules[parent1].is_end = False

        # second conv
        in_ch = modules[parent2].out_features
        block2 = ConvBlock(in_ch, out_channels, is_end=False)

        edges.append((parent2, len(modules)))
        modules.append(block2)
        modules[parent2].is_end = False

        # aggregation conv
        out_ch = 2 ** np.random.randint(3, 6)
        sum_block = ConvBlock(out_channels, out_ch)

        edges.extend([
            (len(modules)-1, len(modules)),
            (len(modules)-2, len(modules))
        ])

        modules.append(sum_block)
    
    end_idxs = []
    for i, module in enumerate(modules):
        if module.is_end:
            preend_block = FlattenBlock(module.out_features, 28, is_end=False)
            end_block = LinearBlock(preend_block.out_features, 20, is_end=False)
            end_idxs.append(len(modules)+1)
            edges.append((i, len(modules)))
            edges.append((len(modules), len(modules)+1))
            modules.append(preend_block)
            modules.append(end_block)
            module.is_end = False
    
    # Creating the final classifier
    classifier = nn.Linear(20, 10)
    edges.extend([
        (end_idx, len(modules)) for end_idx in end_idxs
        ])
    modules.append(classifier)

    return modules, edges