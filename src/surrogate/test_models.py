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

def test_models(name2model, dataset_test, device):
    """
    Caclulate RMSE for surrogate model. Returns dicts: model_name -> metric, model_name -> preds
    """
    name2preds = defaultdict(list)
    y_true = defaultdict(list)

    for name_model, model in name2model.items():
        if isinstance(model,
                      sklearn.linear_model._base.LinearRegression):
            preds = model.predict(dataset_test.tensors[0].numpy())
            name2preds[name_model] = preds
        
        elif isinstance(model, torch.Tensor):
            for x, _ in dataset_test:
                x = x.to(device).bool()
                pred = model[x].mean()
                name2preds[name_model].append(pred.item())

        else:
            with torch.no_grad():
                model.eval()
                model.to(device)
                for x, _ in dataset_test:
                    x = x.to(device)
                    pred = model(x.unsqueeze(0))
                    name2preds[name_model].append(pred.item())

    name2metric = dict()
    y_true = dataset_test.tensors[1]
    for name_model, preds in name2preds.items():
        name2metric[name_model] = np.sqrt(mse(y_true, preds)) * 100

    return name2metric, name2preds, y_true


def collect_model_stats(graph_model, test_loader, criterion, optimizer, device):
    """
    Collect stats for L1- and Molchanov pruning. Returns dict with edge importances.
    """

    graph_model.eval()
    graph_model.reset_edge_stats()

    edge_mask = torch.ones(len(graph_model.edges), device=device)

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = graph_model(x, edge_mask)
        loss = criterion(logits, y)
        loss.backward()

        graph_model.accumulate_importance()
    
    return {
        'l1': graph_model.edge_l1_norms,
        'molchanov': graph_model.edge_molchanov_importance
    }