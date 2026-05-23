from __future__ import annotations

from typing import Any, List, Tuple
from collections import defaultdict

import torch
from torch import nn


class SurrogateModel(nn.Module):
    def __init__(self, edges: List[Tuple[int, int]], is_poly: bool = False):
        super().__init__()

        self.edges = edges
        self.num_edges = len(edges)

        # Edge weights
        self.gamma = nn.Parameter(torch.zeros(self.num_edges))

        self.is_poly = is_poly
        if is_poly:
            self.poly_weights = nn.Parameter(torch.ones(3))

        # Incoming and outgoing edges for each vertex
        incoming = defaultdict(list)
        outgoing = defaultdict(list)
        for idx, (u, v) in enumerate(edges):
            incoming[v].append(idx)
            outgoing[u].append(idx)
        self.incoming = dict(incoming)
        self.outgoing = dict(outgoing)

        # All vertices
        vertices = set()
        for u, v in edges:
            vertices.add(u)
            vertices.add(v)
        self.vertices = sorted(vertices)

        # Out vertex
        self.output_vertex = max(v for v in vertices if v >= 0)
    
    def normalized_edge_weights(self) -> torch.Tensor:
        """
        Returns: Tensor [E], >0 and sum of outgoing edges per vertex == 1
        """
        weights = torch.zeros_like(self.gamma)

        for _, edge_indices in self.outgoing.items():
            logits = self.gamma[edge_indices]
            weights[edge_indices] = torch.softmax(logits, dim=0)

        return weights

    def forward(self, edge_mask: torch.Tensor) -> torch.Tensor:
        """
        edge_mask: Tensor [B, num_edges], values in [0, 1]
        return: Tensor [B]
        """
        assert edge_mask.ndim == 2
        assert edge_mask.shape[1] == self.num_edges

        B = edge_mask.shape[0]
        device = edge_mask.device

        # Normalized edge weights
        edge_weights = self.normalized_edge_weights()  # [E]

        # Values in vertices: vertex -> Tensor [B]
        values_1st = {}
        if self.is_poly:
            values_sq = {}

        # Fixed input
        values_1st[-1] = torch.ones(B, device=device)
        if self.is_poly:
            values_sq[-1] = torch.ones(B, device=device)

        # Topological pass
        for v in self.vertices:
            if v == -1:
                continue

            incoming_edges = self.incoming.get(v, [])
            if not incoming_edges:
                values_1st[v] = torch.zeros(B, device=device)
                if self.is_poly:
                    values_sq[v] = torch.zeros(B, device=device)
                continue

            total_1st = torch.zeros(B, device=device)
            if self.is_poly:
                total_sq = torch.zeros(B, device=device)
            for e_idx in incoming_edges:
                u, _ = self.edges[e_idx]

                edge_val = edge_mask[:, e_idx] * edge_weights[e_idx]

                total_1st = total_1st + (edge_val * values_1st[u])
                if self.is_poly:
                    total_sq = total_sq + ((edge_val ** 2) * values_sq[u])

            values_1st[v] = total_1st
            if self.is_poly:
                values_sq[v] = total_sq

        out_1st = values_1st[self.output_vertex]
        if self.is_poly:
            out_sq = values_sq[self.output_vertex]

            # Вычисляем перекрестные члены (коадаптацию параллельных веток)
            out_cross = (out_1st ** 2) - out_sq

            # Взвешиваем все три смысловых компонента
            w_lin, w_sq, w_cross = self.poly_weights
            output = (w_lin * out_1st) + (w_sq * out_sq) + (w_cross * out_cross)
        else:
            output = out_1st
        return output

    def export_to_uml(self, modules: List[Any], gamma_values: torch.Tensor) -> str:
        """
        Generate PlantUML component diagram from (modules, edges).
        gamma_values: Tensor [E] — edge weight values (e.g. normalized_edge_weights())
        """
        lines = []
        lines.append("@startuml")
        lines.append("skinparam componentStyle rectangle")
        lines.append("left to right direction")
        lines.append("")

        # --- Input node ---
        lines.append('component "Input -1" as N_minus1')

        # --- Modules ---
        for idx, module in enumerate(modules):
            if hasattr(module, "linear"):
                label = (
                    f"LinearBlock ({idx})\\n"
                    f"{module.linear.in_features} → {module.linear.out_features}"
                )
            elif module.__class__.__name__ == "IdBlock":
                label = f"IdBlock (0)"
            else:
                label = f"{type(module).__name__} ({idx})"

            lines.append(f'component "{label}" as N_{idx}')

        lines.append("")

        # --- Edges WITH gamma ---
        for e_idx, (i, j) in enumerate(self.edges):
            src = "N_minus1" if i == -1 else f"N_{i}"
            dst = f"N_{j}"

            gamma_val = float(gamma_values[e_idx])
            gamma_str = f"{gamma_val:.3f}"

            lines.append(f"{src} --> {dst} : {gamma_str}")

        lines.append("")
        lines.append("@enduml")

        return "\n".join(lines)
