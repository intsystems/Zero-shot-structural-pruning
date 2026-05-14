from __future__ import annotations

import os
from datetime import datetime

import torch
from torch import nn
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from tqdm import trange


def update_log(idx, filename):
    timestamp = str(datetime.now())
    os.makedirs("logs", exist_ok=True)
    filepath = f'logs/{filename}.log'

    with open(filepath, 'w') as f:
        f.write(f'{timestamp}\t{idx+1}\t{filename}\n')

def get_data(batch_size=64, data_dir="./data", num_workers=8):
    """
    Create train and test MNIST loaders.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

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
