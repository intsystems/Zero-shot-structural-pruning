"""Graph surrogate model that predicts loss from a per-group binary mask.

Vertices index the pruning groups (0..n_groups-1) plus a virtual input vertex
(-1, always alive) and a virtual output vertex (n_groups, ungated). Edges are
learnable via `gamma`; softmax over outgoing edges of each source vertex
yields non-negative weights summing to 1 per source.

Forward semantics (per batch element):
    value(input) = 1
    value(v) = mask[v] * sum_{u -> v} softmax(gamma)_{u -> v} * value(u)
where `mask[v] = 1` for the input vertex and for the output vertex (it is
ungated by construction).

The output is the surrogate's prediction of `1 - normalized_loss` for the
given per-group binary mask. Per-group importance is later derived by
leave-one-out ablation on this surrogate.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

import torch
from torch import nn


class SurrogateModel(nn.Module):
    def __init__(self, edges: List[Tuple[int, int]], n_groups: int):
        super().__init__()
        self.edges = edges
        self.num_edges = len(edges)
        self.n_groups = n_groups
        self.output_vertex = n_groups
        self.input_vertex = -1

        self.gamma = nn.Parameter(torch.zeros(self.num_edges))

        incoming = defaultdict(list)
        outgoing = defaultdict(list)
        for idx, (u, v) in enumerate(edges):
            incoming[v].append(idx)
            outgoing[u].append(idx)
        self.incoming = dict(incoming)
        self.outgoing = dict(outgoing)

        vertices = {self.input_vertex, self.output_vertex}
        for u, v in edges:
            vertices.add(u)
            vertices.add(v)
        self.vertices = _topological_order(vertices, edges, self.input_vertex)

    def normalized_edge_weights(self) -> torch.Tensor:
        """Softmax over outgoing edges per source vertex — sums to 1 per source."""
        weights = torch.zeros_like(self.gamma)
        for edge_indices in self.outgoing.values():
            logits = self.gamma[edge_indices]
            weights[edge_indices] = torch.softmax(logits, dim=0)
        return weights

    def forward(self, vertex_mask: torch.Tensor) -> torch.Tensor:
        """vertex_mask: [B, n_groups] with values in [0, 1]. Returns [B]."""
        assert vertex_mask.ndim == 2 and vertex_mask.shape[1] == self.n_groups

        B = vertex_mask.shape[0]
        device = vertex_mask.device
        edge_weights = self.normalized_edge_weights()

        values = {self.input_vertex: torch.ones(B, device=device)}
        ones = torch.ones(B, device=device)

        for v in self.vertices:
            if v == self.input_vertex:
                continue
            incoming_edges = self.incoming.get(v, [])
            if not incoming_edges:
                values[v] = torch.zeros(B, device=device)
                continue
            total = torch.zeros(B, device=device)
            for e_idx in incoming_edges:
                u, _ = self.edges[e_idx]
                total = total + edge_weights[e_idx] * values[u]
            gate = ones if v == self.output_vertex else vertex_mask[:, v]
            values[v] = gate * total

        return values[self.output_vertex]


def _topological_order(vertices, edges, input_vertex):
    """Kahn's algorithm on the group-level DAG, with input_vertex first."""
    from collections import defaultdict, deque

    indeg = {v: 0 for v in vertices}
    succ = defaultdict(list)
    for u, v in edges:
        succ[u].append(v)
        indeg[v] = indeg.get(v, 0) + 1

    ordered = []
    ready = deque(v for v in vertices if indeg[v] == 0)
    # ensure input_vertex leads if it is among sources
    if input_vertex in ready:
        ready.remove(input_vertex)
        ready.appendleft(input_vertex)
    while ready:
        v = ready.popleft()
        ordered.append(v)
        for w in succ[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                ready.append(w)

    if len(ordered) != len(vertices):
        # Cycle or disconnected fragment — fall back to numeric order for
        # unresolved vertices to avoid crashing outright.
        seen = set(ordered)
        ordered.extend(sorted(v for v in vertices if v not in seen))
    return ordered
