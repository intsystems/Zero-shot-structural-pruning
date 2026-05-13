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

def update_log(idx, filename):
    timestamp = str(datetime.now())
    os.makedirs("logs", exist_ok=True)
    filepath = f'logs/{filename}.log'

    with open(filepath, 'w') as f:
        f.write(f'{timestamp}\t{idx+1}\t{filename}\n')

def get_data(batch_size=64):
    """
    Create train and test MNIST loaders.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, test_loader


def train(model, train_loader, criterion, optimizer, device, n_epochs, tqdm_desc, is_surrogate=False):
    """
    Train model. Returns epoch loss history.
    """

    model.train()
    model.to(device)
    losses_history = []
    pbar = trange(n_epochs, desc=tqdm_desc, leave=False)

    if not is_surrogate:
        edge_mask = torch.ones(len(model.edges), device=device)

    for i in pbar:
        update_log(i, 'train')
        
        total_loss = 0.0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            if not is_surrogate:
                logits = model(x, edge_mask)
            else:
                logits = model(x)
            
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += y.size(0)

        epoch_loss = total_loss / total
        losses_history.append(epoch_loss)
        pbar.set_description_str(f'{tqdm_desc}, loss={epoch_loss:.5f}')
    
    return losses_history