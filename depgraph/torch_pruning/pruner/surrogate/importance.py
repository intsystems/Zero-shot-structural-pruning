"""Zero-shot structural importance via a graph surrogate over pruning groups.

For every pruning group defined by the DependencyGraph, a per-group binary mask
is applied to the target model's forward pass (via forward hooks on the root
modules). Masks are enumerated systematically — all-ones, all-zeros, and one
mask per group where only that group is dropped — and the resulting loss on a
user-provided data_loader is recorded. A small graph surrogate (SurrogateModel)
is then trained to reproduce these (mask -> normalized loss) pairs. Per-group
importance is derived by leave-one-out ablation on the trained surrogate.

Because the surrogate produces a *scalar* importance per group, the
per-channel tensor returned by `__call__` is constant within a group — all
channels of a group receive the same score. This method is therefore intended
to be used with `global_pruning=True`, where groups are ranked against each
other.

Example:

    import torch_pruning as tp

    imp = tp.importance.SurrogateImportance(
        surrogate_epochs=1500,
        surrogate_lr=1e-3,
        surrogate_batch_size=32,
    )
    pruner = tp.MetaPruner(
        model, example_inputs,
        importance=imp,
        global_pruning=True,
        pruning_ratio=0.5,
    )
    imp.fit(pruner, data_loader, criterion, device='cuda')
    pruner.step()
"""

from __future__ import annotations

import typing
import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..importance import Importance
from .dataset import build_systematic_mask_dataset
from .model import SurrogateModel


class SurrogateImportance(Importance):
    def __init__(
        self,
        surrogate_epochs: int = 3000,
        surrogate_lr: float = 1e-3,
        surrogate_batch_size: int = 32,
        normalizer: typing.Optional[str] = "mean",
        target_types: typing.Sequence = (
            nn.modules.conv._ConvNd,
            nn.Linear,
            nn.modules.batchnorm._BatchNorm,
        ),
    ):
        self.surrogate_epochs = surrogate_epochs
        self.surrogate_lr = surrogate_lr
        self.surrogate_batch_size = surrogate_batch_size
        self.normalizer = normalizer
        self.target_types = tuple(target_types)
        self._imp: typing.Dict[typing.Tuple[int, typing.Callable], float] = {}

    def fit(self, pruner, data_loader, criterion, device=None) -> None:
        """Sample systematic masks, train the surrogate, cache per-group scores."""
        model = pruner.model
        if device is None:
            device = next(model.parameters()).device

        # 1. Enumerate groups; keep those whose root module is a prunable target.
        groups = list(pruner.DG.get_all_groups(
            ignored_layers=pruner.ignored_layers,
            root_module_types=pruner.root_module_types,
        ))
        root_modules: typing.List[nn.Module] = []
        kept_group_idx: typing.List[int] = []
        for gi, group in enumerate(groups):
            root = group[0].dep.target.module
            if isinstance(root, self.target_types):
                root_modules.append(root)
                kept_group_idx.append(gi)

        if not root_modules:
            raise RuntimeError(
                "SurrogateImportance.fit(): no prunable groups found. "
                "Check target_types and pruner.root_module_types."
            )
        n_groups = len(root_modules)

        # 2. Build a group-level DAG (vertices = groups, edges = data flow).
        edges = _build_group_graph(pruner.DG, root_modules)

        # 3. Install per-group forward hooks that scale root outputs by mask_state.
        mask_state = torch.ones(n_groups, device=device)
        hooks = []
        for i, m in enumerate(root_modules):
            hooks.append(m.register_forward_hook(_make_mask_hook(mask_state, i)))

        try:
            # 4. Build the systematic (mask, loss) dataset.
            dataset = build_systematic_mask_dataset(
                model, n_groups, mask_state, data_loader, criterion, device,
            )
        finally:
            for h in hooks:
                h.remove()

        # 5. Train the graph surrogate on MSE.
        surrogate = SurrogateModel(edges, n_groups=n_groups).to(device)
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=self.surrogate_lr)
        loader = DataLoader(
            dataset,
            batch_size=self.surrogate_batch_size,
            shuffle=True,
        )
        surrogate.train()
        for _ in range(self.surrogate_epochs):
            for m, y in loader:
                m = m.to(device)
                y = y.to(device)
                pred = surrogate(m)
                loss = nn.functional.mse_loss(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 6. Per-group importance = surrogate(1) - surrogate(1 with group i off).
        surrogate.eval()
        with torch.no_grad():
            ones = torch.ones(1, n_groups, device=device)
            baseline = surrogate(ones).item()
            scores = torch.zeros(n_groups, device=device)
            for i in range(n_groups):
                m = ones.clone()
                m[0, i] = 0.0
                dropped = surrogate(m).item()
                scores[i] = baseline - dropped

        # Clip tiny negatives that come from surrogate noise; they'd break
        # ratio-based normalizers.
        scores = scores.clamp(min=0.0)
        scores = _normalize_scores(scores, self.normalizer)

        # 7. Cache scalars keyed by (root_module identity, pruning fn).
        self._imp = {}
        for i, gi in enumerate(kept_group_idx):
            root = root_modules[i]
            root_fn = groups[gi][0].dep.handler
            self._imp[(id(root), root_fn)] = float(scores[i].item())

    @torch.no_grad()
    def __call__(self, group) -> typing.Optional[torch.Tensor]:
        if not self._imp:
            warnings.warn(
                "SurrogateImportance was called before .fit(); returning None."
            )
            return None
        root = group[0].dep.target.module
        fn = group[0].dep.handler
        key = (id(root), fn)
        if key not in self._imp:
            return None
        _, idxs = group[0]
        n = len(idxs)
        # The surrogate produces one score per group, so all channels of the
        # group get the same score. Add a tiny magnitude-based tie-breaker so
        # MetaPruner can pick a subset of channels within the group instead of
        # facing a full tie (which its check_pruning_group refuses to resolve).
        scalar = self._imp[key]
        tie_break = 1e-6 * _channel_weight_norms(root, fn, idxs)
        return torch.full((n,), scalar) + tie_break


def _make_mask_hook(mask_state: torch.Tensor, idx: int):
    def hook(module, input, output):
        return output * mask_state[idx]
    return hook


def _channel_weight_norms(layer: nn.Module, pruning_fn, idxs) -> torch.Tensor:
    """Per-channel L1 magnitudes used purely to break ties inside a group."""
    from ..function import (
        prune_conv_out_channels,
        prune_linear_out_channels,
        prune_batchnorm_out_channels,
    )

    if not hasattr(layer, "weight") or layer.weight is None:
        return torch.zeros(len(idxs))

    w = layer.weight.data
    if pruning_fn in (prune_conv_out_channels, prune_linear_out_channels):
        if hasattr(layer, "transposed") and layer.transposed:
            per_channel = w.transpose(0, 1).flatten(1).abs().sum(1)
        else:
            per_channel = w.flatten(1).abs().sum(1)
    elif pruning_fn == prune_batchnorm_out_channels:
        per_channel = w.abs()
    else:
        return torch.zeros(len(idxs))

    idxs_t = torch.tensor(list(idxs), device=per_channel.device, dtype=torch.long)
    idxs_t = idxs_t.clamp(max=per_channel.numel() - 1)
    per_channel = per_channel[idxs_t]
    denom = per_channel.max() + 1e-12
    return (per_channel / denom).cpu()


def _build_group_graph(DG, root_modules: typing.Sequence[nn.Module]):
    """Build (u, v) edges over groups; -1 = INPUT vertex, len(groups) = OUTPUT."""
    module_to_gidx = {m: i for i, m in enumerate(root_modules)}
    INPUT = -1
    OUTPUT = len(root_modules)

    edges: typing.List[typing.Tuple[int, int]] = []
    for j, m_j in enumerate(root_modules):
        node_j = DG.module2node.get(m_j)
        if node_j is None:
            edges.append((INPUT, j))
            continue
        ancestor_gidxs = _bfs_upstream_boundary(node_j, module_to_gidx)
        if ancestor_gidxs:
            for u in ancestor_gidxs:
                edges.append((u, j))
        else:
            edges.append((INPUT, j))

    sources = {u for (u, _) in edges}
    for j in range(len(root_modules)):
        if j not in sources:
            edges.append((j, OUTPUT))

    # Guarantee the OUTPUT vertex has at least one incoming edge (e.g. if the
    # only group is a single terminal one). Otherwise SurrogateModel would
    # collapse to zero.
    if all(v != OUTPUT for (_, v) in edges):
        edges.append((len(root_modules) - 1, OUTPUT))

    return edges


def _bfs_upstream_boundary(start_node, module_to_gidx) -> typing.Set[int]:
    """BFS on Node.inputs; collect gidxs of the first-hit root-module ancestors."""
    visited = {id(start_node)}
    stack = list(start_node.inputs)
    boundary: typing.Set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        gidx = module_to_gidx.get(node.module)
        if gidx is not None:
            boundary.add(gidx)
        else:
            stack.extend(node.inputs)
    return boundary


def _normalize_scores(scores: torch.Tensor, normalizer: typing.Optional[str]) -> torch.Tensor:
    if normalizer is None:
        return scores
    if normalizer == "mean":
        return scores / (scores.mean() + 1e-8)
    if normalizer == "sum":
        return scores / (scores.sum() + 1e-8)
    if normalizer == "max":
        return scores / (scores.max() + 1e-8)
    if normalizer == "standardization":
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    if normalizer == "gaussian":
        return (scores - scores.mean()) / (scores.std() + 1e-8)
    raise ValueError(f"Unknown normalizer: {normalizer!r}")
