from __future__ import annotations

from collections import defaultdict

import torch
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error as mse


def get_expected_intersection(y_true_top: np.ndarray, preds: np.ndarray, K: int) -> float:
    """
    Expected size of the intersection between the true top-K set and the predicted top-K set,
    averaged over all possible tie-breaking orderings.
    """
    # Finding threshold
    threshold = np.partition(preds, -K)[-K]

    idx_greater = np.where(preds > threshold)[0]  # keep
    idx_equal = np.where(preds == threshold)[0]    # for sampling

    needed = K - len(idx_greater)

    # Intersects
    direct_intersect = np.intersect1d(y_true_top, idx_greater).shape[0]
    equal_intersect = np.intersect1d(y_true_top, idx_equal).shape[0]

    # Expected mean of intersects
    expected_intersect = direct_intersect + (equal_intersect * (needed / len(idx_equal)))

    return expected_intersect


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
