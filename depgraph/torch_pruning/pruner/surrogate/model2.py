from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

import torch
from torch import nn


class ChannelSurrogateModel(nn.Module):
    def __init__(
        self,
        edges: List[Tuple[int, int]],
        n_groups: int,
        group_counts: List[int],
    ):
        super().__init__()
        self.edges = edges
        self.num_edges = len(edges)
        self.n_groups = n_groups
        self.output_vertex = n_groups
        self.input_vertex = -1

        assert len(group_counts) == n_groups, (
            f"Expected {n_groups} group counts, got {len(group_counts)}"
        )
        self.group_counts = group_counts
        self.total_channels = sum(group_counts)

        # Предрасчет смещений каналов (offsets) для каждой группы 0..n_groups-1
        self.offsets = []
        curr = 0
        for count in group_counts:
            self.offsets.append((curr, curr + count))
            curr += count

        # 1. Веса ребер графа (между группами)
        self.gamma = nn.Parameter(torch.zeros(self.num_edges))

        # 2. ИЕРАРХИЯ - Уровень 1: Вес важности ВСЕЙ ГРУППЫ (послойный гейт)
        self.group_importance = nn.Parameter(torch.zeros(self.n_groups))

        # 3. ИЕРАРХИЯ - Уровень 2: Веса КАНАЛОВ ВНУТРИ ГРУПП (поканальный гейт)
        self.channel_weights = nn.Parameter(torch.zeros(self.total_channels))

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

    def _get_group_gate(self, v: int, channel_mask: torch.Tensor) -> torch.Tensor:
        """
        Иерархический гейт:
        Gate(v) = Layer_Gate(v) * Channel_Gate(v)
        """
        # --- Уровень 1: Послойный гейт (Layer-level Gate) ---
        # Sigmoid переводит вес группы в пропускную способность [0, 1]
        layer_gate = torch.sigmoid(self.group_importance[v])

        # --- Уровень 2: Поканальный гейт (Channel-level Gate) ---
        start, end = self.offsets[v]
        c_mask = channel_mask[:, start:end]  # [B, C_v]
        
        # Нормализованные относительные веса каналов внутри этой группы: [C_v]
        c_weights = torch.softmax(self.channel_weights[start:end], dim=0)
        
        # Взвешенная сумма активных каналов: [B]
        channel_gate = torch.sum(c_mask * c_weights.unsqueeze(0), dim=-1)

        # --- Иерархическая комбинация ---
        # Пропускная способность = (Важность слоя) * (Сохраненная доля поканальной емкости)
        return layer_gate * channel_gate

    def forward(self, channel_mask: torch.Tensor) -> torch.Tensor:
        """
        channel_mask: [B, total_channels] со значениями в [0, 1].
        Returns: [B] — предсказанная пропускная способность графа (1 - normalized_loss).
        """
        assert channel_mask.ndim == 2 and channel_mask.shape[1] == self.total_channels, (
            f"Expected channel_mask shape [B, {self.total_channels}], got {channel_mask.shape}"
        )

        B = channel_mask.shape[0]
        device = channel_mask.device
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

            # Агрегация входящих потоков от предыдущих слоев
            total = torch.zeros(B, device=device)
            for e_idx in incoming_edges:
                u, _ = self.edges[e_idx]
                total = total + edge_weights[e_idx] * values[u]

            # Вычисление иерархического гейта для текущей вершины v
            if v == self.output_vertex:
                gate = ones
            else:
                gate = self._get_group_gate(v, channel_mask)

            # Передача сигнала далее по графу
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
        seen = set(ordered)
        ordered.extend(sorted(v for v in vertices if v not in seen))
    return ordered