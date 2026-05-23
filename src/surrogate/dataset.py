from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from tqdm import trange

from utils import update_log


def create_surrogate_dataset(graph_model, test_loader, criterion, n_iterations, mask_prob, device):
    """
    Create a dataset for surrogate model. Returns train and test Torch datasets.
    """

    graph_model.eval()

    n_edges = len(graph_model.edges)
    masks_train = [torch.ones(n_edges), torch.zeros(n_edges)]
    losses_train = []
    masks_set = set([0, 2**n_edges-1])

    for edge_mask in masks_train:
        loss_mask = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = graph_model(x, edge_mask)
                loss = criterion(logits, y)
                loss_mask += loss.item()
                total += y.size(0)
        losses_train.append(loss_mask / total)
    
    # Iterate masks and calculate loss
    tbar = trange(n_edges, desc='Train masks', leave=False)
    for zero_idx in tbar:
        update_log(zero_idx, 'create_surrogate_dataset_train')

        edge_mask = torch.ones(n_edges)
        edge_mask[zero_idx] = 0
        
        edge_mask_int = int(''.join(map(lambda x: str(x.int().item()), edge_mask)), 2)
        masks_set.add(edge_mask_int)
        
        loss_mask = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = graph_model(x, edge_mask)
                loss = criterion(logits, y)
                loss_mask += loss.item()
                total += y.size(0)

        masks_train.append(edge_mask)
        losses_train.append(loss_mask / total)
    
    # Normalize losses to [0, 1] and invert
    losses_train =  torch.tensor(losses_train)
    losses_train = (losses_train - losses_train.min()) / (1e-10 + losses_train.max() - losses_train.min())
    losses_train = 1 - losses_train

    masks_test = []
    losses_test = []
    
    # Sampling masks and calculate loss
    tbar = trange(n_iterations, desc='Test masks', leave=False)
    for i in tbar:
        update_log(i, 'create_surrogate_dataset_test')
        
        edge_mask_int = -1
        while edge_mask_int in masks_set:
            edge_mask = (torch.rand((n_edges,)) >= mask_prob).float()
            edge_mask_int = int(''.join(map(lambda x: str(x.int().item()), edge_mask)), 2)
        masks_set.add(edge_mask_int)
        
        loss_mask = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = graph_model(x, edge_mask)
                loss = criterion(logits, y)
                loss_mask += loss.item()
                total += y.size(0)

        masks_test.append(edge_mask)
        losses_test.append(loss_mask / total)
    
    # Normalize losses to [0, 1] and invert
    losses_test =  torch.tensor(losses_test)
    losses_test = (losses_test - losses_test.min()) / (1e-10 + losses_test.max() - losses_test.min())
    losses_test = 1 - losses_test

    # Сreating datasets
    dataset_train = TensorDataset(torch.vstack(masks_train), losses_train)
    dataset_test = TensorDataset(torch.vstack(masks_test), losses_test)
    
    return dataset_train, dataset_test
