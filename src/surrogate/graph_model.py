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

class SurrogateModel(nn.Module):
    def __init__(self, edges: List[Tuple[int, int]], is_poly: bool = False):
        super().__init__()

        self.edges = edges
        self.num_edges = len(edges)

        # Edge weights
        self.gamma = nn.Parameter(torch.zeros(self.num_edges))

        self.is_poly = is_poly
        if is_poly:
            self.poly_weights = nn.Parameter(torch.ones(3))

        # Incoming and outgoing edges for each vertex
        incoming = defaultdict(list)
        outgoing = defaultdict(list)
        for idx, (u, v) in enumerate(edges):
            incoming[v].append(idx)
            outgoing[u].append(idx)
        self.incoming = dict(incoming)
        self.outgoing = dict(outgoing)

        # All vertices
        vertices = set()
        for u, v in edges:
            vertices.add(u)
            vertices.add(v)
        self.vertices = sorted(vertices)

        # Out vertex
        self.output_vertex = max(v for v in vertices if v >= 0)
    
    def normalized_edge_weights(self) -> torch.Tensor:
        """
        Returns: Tensor [E], >0 and sum of outgoing edges per vertex == 1
        """
        weights = torch.zeros_like(self.gamma)

        for _, edge_indices in self.outgoing.items():
            logits = self.gamma[edge_indices]
            weights[edge_indices] = torch.softmax(logits, dim=0)

        return weights

    def forward(self, edge_mask: torch.Tensor) -> torch.Tensor:
        """
        edge_mask: Tensor [B, num_edges], values in [0, 1]
        return: Tensor [B]
        """
        assert edge_mask.ndim == 2
        assert edge_mask.shape[1] == self.num_edges

        B = edge_mask.shape[0]
        device = edge_mask.device

        # Normalized edge weights
        edge_weights = self.normalized_edge_weights()  # [E]

        # Values in vertices: vertex -> Tensor [B]
        values_1st = {}
        if self.is_poly:
            values_sq = {}

        # Fixed input
        values_1st[-1] = torch.ones(B, device=device)
        if self.is_poly:
            values_sq[-1] = torch.ones(B, device=device)

        # Topological pass
        for v in self.vertices:
            if v == -1:
                continue

            incoming_edges = self.incoming.get(v, [])
            if not incoming_edges:
                values_1st[v] = torch.zeros(B, device=device)
                if self.is_poly:
                    values_sq[v] = torch.zeros(B, device=device)
                continue

            total_1st = torch.zeros(B, device=device)
            if self.is_poly:
                total_sq = torch.zeros(B, device=device)
            for e_idx in incoming_edges:
                u, _ = self.edges[e_idx]

                edge_val = edge_mask[:, e_idx] * edge_weights[e_idx]

                total_1st = total_1st + (edge_val * values_1st[u])
                if self.is_poly:
                    total_sq = total_sq + ((edge_val ** 2) * values_sq[u])

            values_1st[v] = total_1st
            if self.is_poly:
                values_sq[v] = total_sq

        out_1st = values_1st[self.output_vertex]
        if self.is_poly:
            out_sq = values_sq[self.output_vertex]

            # Вычисляем перекрестные члены (коадаптацию параллельных веток)
            out_cross = (out_1st ** 2) - out_sq

            # Взвешиваем все три смысловых компонента
            w_lin, w_sq, w_cross = self.poly_weights
            output = (w_lin * out_1st) + (w_sq * out_sq) + (w_cross * out_cross)
        else:
            output = out_1st
        return output