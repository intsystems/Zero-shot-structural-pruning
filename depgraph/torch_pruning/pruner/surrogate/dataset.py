"""Systematic mask sampling and loss evaluation for the surrogate."""

from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import torch
from torch.utils.data import TensorDataset


@torch.no_grad()
def build_systematic_mask_dataset(
    model: torch.nn.Module,
    n_groups: int,
    mask_state: torch.Tensor,
    data_loader: Iterable,
    criterion: Callable,
    device: str | torch.device,
) -> TensorDataset:
    """Build a (masks, losses) dataset by systematically toggling one group off at a time.

    Masks:
        - all-ones
        - all-zeros
        - one all-ones-with-position-k-zero for each k in [0, n_groups)

    For every mask the loss is averaged over the whole data_loader; the resulting
    vector is min-max normalized to [0, 1] and inverted (1 - x) so that a higher
    score means "this configuration hurts more when removed".

    Assumes that the caller has installed forward hooks on the model that read
    values from `mask_state` (a tensor of shape [n_groups] on `device`) — this
    function just writes into `mask_state` and runs forward passes.
    """
    model.eval()

    masks: List[torch.Tensor] = [torch.ones(n_groups), torch.zeros(n_groups)]
    for k in range(n_groups):
        m = torch.ones(n_groups)
        m[k] = 0.0
        masks.append(m)

    losses: List[float] = []
    for mask in masks:
        mask_state.copy_(mask.to(mask_state.device))
        total_loss = 0.0
        total = 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            total += y.size(0)
        losses.append(total_loss / max(total, 1))

    losses_t = torch.tensor(losses)
    losses_t = (losses_t - losses_t.min()) / (1e-10 + losses_t.max() - losses_t.min())
    losses_t = 1.0 - losses_t

    return TensorDataset(torch.vstack(masks), losses_t)
