"""Shared helpers for SurrogateImportance experiments on CIFAR-10.

Every experiment in this folder trains a small model briefly, measures baseline
accuracy, prunes with a given Importance, measures again, fine-tunes for a
couple of epochs, and reports the final numbers. This module centralizes the
data pipeline, the train/eval loops, and the CLI so each experiment file only
needs to describe its model and the pruner configuration.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
import typing

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision import transforms

# Ensure the repository root (containing `torch_pruning/`) is importable when
# scripts are run directly with `python exp*.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch_pruning as tp  # noqa: E402


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Per-family training defaults. "conv" covers plain/residual CNNs (VGG,
# ResNet, MobileNetV2, ConvMixer); "vit" covers transformer blocks, which
# need an adaptive optimizer, higher weight decay, and more epochs to be
# comparable to CNNs trained with plain SGD.
MODEL_HPARAMS: typing.Dict[str, dict] = {
    "conv": dict(optimizer="sgd", lr=1e-2, weight_decay=5e-4,
                 epochs=5, finetune_lr=1e-3),
    "vit": dict(optimizer="adamw", lr=1e-3, weight_decay=0.05,
                epochs=10, finetune_lr=1e-4),
}


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, torch (and numpy if present) RNGs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_cifar10_loaders(data_dir: str, batch_size: int, num_workers: int):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_ds = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    calib_ds = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, calib_ds


def calibration_loader(calib_ds, n_samples: int, batch_size: int,
                       num_classes: int = 10) -> DataLoader:
    """Stratified, shuffled calibration subset.

    CIFAR-10 train data is ordered by class, so taking the first N samples
    yields a single-class subset (all "airplane" for N <= 5000). Instead,
    take `n_samples // num_classes` images per class and shuffle them.
    """
    targets = calib_ds.targets
    per_class = max(1, n_samples // num_classes)
    idx_by_class: typing.Dict[int, typing.List[int]] = {}
    for i, y in enumerate(targets):
        idx_by_class.setdefault(y, []).append(i)
    idx: typing.List[int] = []
    for y in sorted(idx_by_class):
        idx.extend(idx_by_class[y][:per_class])
    rng = random.Random(0)  # fixed seed -> the subset itself is reproducible
    rng.shuffle(idx)
    return DataLoader(Subset(calib_ds, idx), batch_size=batch_size, shuffle=False)


def train_one_epoch(model, loader, opt, criterion, device):
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def count_stats(model, example_inputs):
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    return macs, nparams


def make_common_parser(description: str,
                        family: str = "conv") -> argparse.ArgumentParser:
    """Common CLI with per-family defaults (see MODEL_HPARAMS)."""
    hp = MODEL_HPARAMS[family]
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=hp["epochs"],
                   help="pre-pruning training epochs")
    p.add_argument("--finetune-epochs", type=int, default=2,
                   help="post-pruning fine-tune epochs")
    p.add_argument("--lr", type=float, default=hp["lr"])
    p.add_argument("--finetune-lr", type=float, default=hp["finetune_lr"])
    p.add_argument("--pruning-ratio", type=float, default=0.3)
    p.add_argument("--surrogate-epochs", type=int, default=1000)
    p.add_argument("--surrogate-lr", type=float, default=5e-3)
    p.add_argument("--surrogate-batch-size", type=int, default=32)
    p.add_argument("--calibration-samples", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true",
                   help="torch.cudnn.deterministic (slower, reproducible)")
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default=hp["optimizer"])
    p.add_argument("--weight-decay", type=float, default=hp["weight_decay"])
    return p


def make_optimizer(name: str, model: nn.Module, lr: float, weight_decay: float):
    """Optimizer with weight decay excluded for 1-D params (norms, biases).

    Note: takes the *model*, not a params iterable, so parameters can be
    split into decay / no-decay groups.
    """
    decay, no_decay = [], []
    for m_name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        if prm.ndim <= 1:  # BN/ biases / LayerNorm weights
            no_decay.append(prm)
        else:
            decay.append(prm)
    groups = [
        dict(params=decay, weight_decay=weight_decay),
        dict(params=no_decay, weight_decay=0.0),
    ]
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    return torch.optim.SGD(groups, lr=lr, momentum=0.9)


def accumulate_taylor_gradients(model, calib_loader, criterion, device) -> None:
    """Average Taylor gradients over the calibration subset.

    Each batch loss is scaled by 1/n_batches so the accumulated gradient
    is on the same scale regardless of `--calibration-samples`.
    """
    model.zero_grad()
    n_batches = max(len(calib_loader), 1)
    for x, y in calib_loader:
        x, y = x.to(device), y.to(device)
        (criterion(model(x), y) / n_batches).backward()


def run_pipeline(
    experiment_name: str,
    model_fn: typing.Callable[[], nn.Module],
    args: argparse.Namespace,
    example_inputs_shape: typing.Tuple[int, ...] = (1, 3, 32, 32),
    ignored_layers_fn: typing.Callable[[nn.Module], typing.List[nn.Module]] = None,
    pruner_kwargs: typing.Union[dict, typing.Callable[[nn.Module], dict], None] = None,
    importance: typing.Optional[typing.Callable] = None,
    calibration_needed: bool = True,
) -> dict:
    """Full train → prune → fine-tune → report cycle.

    `importance=None` means SurrogateImportance (built from CLI args) and
    `calibration_needed=True` triggers `imp.fit(...)`. Passing an existing
    Importance skips those steps — useful for method-comparison sweeps.
    """
    set_seed(args.seed, deterministic=getattr(args, "deterministic", False))
    device = torch.device(args.device)
    print(f"\n########## {experiment_name} ##########")
    print(f"Device: {device}  seed={args.seed}")

    train_loader, test_loader, calib_ds = get_cifar10_loaders(
        args.data_dir, args.batch_size, args.num_workers)
    calib_loader = calibration_loader(calib_ds, args.calibration_samples, args.batch_size)

    model = model_fn().to(device)
    example_inputs = torch.randn(*example_inputs_shape, device=device)
    criterion = nn.CrossEntropyLoss()

    # ---- baseline train ----
    opt = make_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    print(f"[baseline train] {args.epochs} epochs, {args.optimizer} "
          f"lr={args.lr} wd={args.weight_decay}")
    for epoch in range(args.epochs):
        t0 = time.time()
        loss, tr_acc = train_one_epoch(model, train_loader, opt, criterion, device)
        scheduler.step()
        print(f"  epoch {epoch+1}/{args.epochs}  loss={loss:.4f}  "
              f"train_acc={tr_acc*100:.2f}%  ({time.time()-t0:.1f}s)")

    base_acc = evaluate(model, test_loader, device)
    base_macs, base_nparams = count_stats(model, example_inputs)
    print(f"[baseline] test_acc={base_acc*100:.2f}%  "
          f"params={base_nparams/1e6:.2f}M  MACs={base_macs/1e9:.2f}G")

    # ---- prune ----
    ignored = ignored_layers_fn(model) if ignored_layers_fn else []
    if importance is None:
        importance = tp.importance.SurrogateImportance(
            surrogate_epochs=args.surrogate_epochs,
            surrogate_lr=args.surrogate_lr,
            surrogate_batch_size=args.surrogate_batch_size,
            normalizer="mean",
        )

    default_pruner_kwargs = dict(
        model=model,
        example_inputs=example_inputs,
        importance=importance,
        global_pruning=True,
        pruning_ratio=args.pruning_ratio,
        ignored_layers=ignored,
    )
    extra = pruner_kwargs(model) if callable(pruner_kwargs) else pruner_kwargs
    if extra:
        default_pruner_kwargs.update(extra)
    pruner = tp.pruner.MetaPruner(**default_pruner_kwargs)

    if calibration_needed and hasattr(importance, "fit"):
        t0 = time.time()
        importance.fit(pruner, calib_loader, criterion, device=device)
        print(f"[fit] {len(getattr(importance, '_imp', {}))} groups "
              f"scored in {time.time()-t0:.1f}s")
    elif isinstance(importance, tp.importance.TaylorImportance):
        # TaylorImportance needs a fresh backward pass to populate gradients.
        accumulate_taylor_gradients(model, calib_loader, criterion, device)
        print(f"[taylor] gradients averaged over {args.calibration_samples} "
              f"stratified samples")

    pruner.step()
    pruned_macs, pruned_nparams = count_stats(model, example_inputs)
    pruned_acc = evaluate(model, test_loader, device)
    print(f"[after prune] test_acc={pruned_acc*100:.2f}%  "
          f"params={pruned_nparams/1e6:.2f}M "
          f"(−{(1-pruned_nparams/base_nparams)*100:.1f}%)  "
          f"MACs={pruned_macs/1e9:.2f}G "
          f"(−{(1-pruned_macs/base_macs)*100:.1f}%)")

    # ---- fine-tune ----
    if args.finetune_epochs > 0:
        opt = make_optimizer(args.optimizer, model, args.finetune_lr, args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(args.finetune_epochs, 1))
        print(f"[finetune] {args.finetune_epochs} epochs lr={args.finetune_lr}")
        for epoch in range(args.finetune_epochs):
            t0 = time.time()
            loss, tr_acc = train_one_epoch(model, train_loader, opt, criterion, device)
            scheduler.step()
            print(f"  epoch {epoch+1}/{args.finetune_epochs}  loss={loss:.4f}  "
                  f"train_acc={tr_acc*100:.2f}%  ({time.time()-t0:.1f}s)")
        ft_acc = evaluate(model, test_loader, device)
    else:
        ft_acc = pruned_acc

    print(f"\n=== {experiment_name}: summary ===")
    print(f"  baseline           acc={base_acc*100:6.2f}%   params={base_nparams/1e6:6.2f}M")
    print(f"  after prune        acc={pruned_acc*100:6.2f}%   params={pruned_nparams/1e6:6.2f}M "
          f"(−{(1-pruned_nparams/base_nparams)*100:.1f}%)")
    print(f"  after fine-tune    acc={ft_acc*100:6.2f}%")
    return dict(
        baseline_acc=base_acc,
        pruned_acc=pruned_acc,
        finetuned_acc=ft_acc,
        base_params=base_nparams,
        pruned_params=pruned_nparams,
        base_macs=base_macs,
        pruned_macs=pruned_macs,
    )
