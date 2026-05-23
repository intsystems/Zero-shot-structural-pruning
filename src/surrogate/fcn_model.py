from __future__ import annotations

from torch import nn


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
