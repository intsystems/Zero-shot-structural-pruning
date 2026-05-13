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

def create_surrogate_fcn_model(n_edges, fcn_dims, add_sigmoid=False):
    """
    Create a surrogate FCN baseline. Returns nn.Module.
    """
    
    layers = [nn.Linear(n_edges, fcn_dims[0]), nn.ReLU()]

    for i in range(1, len(fcn_dims)):
        layers.append(nn.Linear(fcn_dims[i-1], fcn_dims[i]))
        layers.append(nn.ReLU())
    
    layers.append(nn.Linear(fcn_dims[-1], 1))
    if add_sigmoid:
        layers.append(nn.Sigmoid())
    
    return nn.Sequential(*layers)