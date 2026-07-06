"""End-to-end test for SurrogateImportance on ResNet-18 / CIFAR-10.

Trains a CIFAR-adapted ResNet-18 briefly, measures baseline accuracy,
applies structural pruning driven by SurrogateImportance, re-measures
accuracy, then fine-tunes for a couple of epochs and reports the final
accuracy along with parameter/FLOPs reduction.

Usage:
    # quick sanity run (fast, low accuracy, ~few minutes on CPU)
    python test_surrogate_resnet18_cifar.py

    # more realistic evaluation (dozens of minutes on CPU, minutes on GPU)
    python test_surrogate_resnet18_cifar.py \
        --epochs 30 --finetune-epochs 20 --pruning-ratio 0.5

CIFAR-10 will be auto-downloaded to `--data-dir` on first run.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision import transforms
from torchvision.models.resnet import BasicBlock, ResNet

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
import torch_pruning as tp


def build_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """torchvision ResNet-18 adapted for 32x32 CIFAR inputs."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def get_cifar10_loaders(data_dir: str, batch_size: int, num_workers: int):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        data_dir, train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(
        data_dir, train=False, download=True, transform=test_tf)
    calib_ds = torchvision.datasets.CIFAR10(
        data_dir, train=True, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, calib_ds


def calibration_loader(calib_ds, n_samples: int, batch_size: int):
    """Deterministic small subset used by SurrogateImportance.fit()."""
    idx = list(range(min(n_samples, len(calib_ds))))
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
    return correct / total


def count_stats(model, example_inputs):
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    return macs, nparams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3,
                   help="pre-pruning training epochs")
    p.add_argument("--finetune-epochs", type=int, default=2,
                   help="post-pruning fine-tune epochs")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--finetune-lr", type=float, default=1e-3)
    p.add_argument("--pruning-ratio", type=float, default=0.3)
    p.add_argument("--surrogate-epochs", type=int, default=1000)
    p.add_argument("--surrogate-lr", type=float, default=5e-3)
    p.add_argument("--calibration-samples", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")

    train_loader, test_loader, calib_ds = get_cifar10_loaders(
        args.data_dir, args.batch_size, args.num_workers)
    calib_loader = calibration_loader(
        calib_ds, args.calibration_samples, args.batch_size)

    model = build_cifar_resnet18(num_classes=10).to(device)
    example_inputs = torch.randn(1, 3, 32, 32, device=device)
    criterion = nn.CrossEntropyLoss()

    # ---------------- Baseline training ----------------
    opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.epochs, 1))
    print(f"\n=== Baseline training ({args.epochs} epochs) ===")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, opt, criterion, device)
        scheduler.step()
        print(f"  epoch {epoch+1}/{args.epochs}  "
              f"loss={train_loss:.4f}  train_acc={train_acc*100:.2f}%  "
              f"({time.time()-t0:.1f}s)")

    base_acc = evaluate(model, test_loader, device)
    base_macs, base_nparams = count_stats(model, example_inputs)
    print(f"\n[baseline] test_acc={base_acc*100:.2f}%  "
          f"params={base_nparams/1e6:.2f}M  MACs={base_macs/1e9:.2f}G")

    # ---------------- Pruning ----------------
    ignored_layers = [m for m in model.modules()
                      if isinstance(m, nn.Linear) and m.out_features == 10]

    imp = tp.importance.SurrogateImportance(
        surrogate_epochs=args.surrogate_epochs,
        surrogate_lr=args.surrogate_lr,
        surrogate_batch_size=32,
        normalizer="mean",
    )
    pruner = tp.pruner.MetaPruner(
        model, example_inputs,
        importance=imp,
        global_pruning=True,
        pruning_ratio=args.pruning_ratio,
        ignored_layers=ignored_layers,
    )

    print(f"\n=== SurrogateImportance.fit() on "
          f"{args.calibration_samples} calibration samples ===")
    t0 = time.time()
    imp.fit(pruner, calib_loader, criterion, device=device)
    print(f"  fit done in {time.time()-t0:.1f}s, "
          f"cached scores for {len(imp._imp)} groups")

    pruner.step()
    pruned_macs, pruned_nparams = count_stats(model, example_inputs)
    pruned_acc = evaluate(model, test_loader, device)
    print(f"\n[after prune] test_acc={pruned_acc*100:.2f}%  "
          f"params={pruned_nparams/1e6:.2f}M "
          f"(−{(1-pruned_nparams/base_nparams)*100:.1f}%)  "
          f"MACs={pruned_macs/1e9:.2f}G "
          f"(−{(1-pruned_macs/base_macs)*100:.1f}%)")

    # ---------------- Fine-tune ----------------
    if args.finetune_epochs > 0:
        print(f"\n=== Fine-tune ({args.finetune_epochs} epochs) ===")
        opt = torch.optim.SGD(model.parameters(), lr=args.finetune_lr,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(args.finetune_epochs, 1))
        for epoch in range(args.finetune_epochs):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, opt, criterion, device)
            scheduler.step()
            print(f"  epoch {epoch+1}/{args.finetune_epochs}  "
                  f"loss={train_loss:.4f}  train_acc={train_acc*100:.2f}%  "
                  f"({time.time()-t0:.1f}s)")
        ft_acc = evaluate(model, test_loader, device)
        print(f"\n[after finetune] test_acc={ft_acc*100:.2f}%")
    else:
        ft_acc = pruned_acc

    # ---------------- Summary ----------------
    print("\n=== Summary ===")
    print(f"  baseline           acc = {base_acc*100:.2f}%   "
          f"params = {base_nparams/1e6:.2f}M")
    print(f"  after prune        acc = {pruned_acc*100:.2f}%   "
          f"params = {pruned_nparams/1e6:.2f}M "
          f"(−{(1-pruned_nparams/base_nparams)*100:.1f}%)")
    print(f"  after finetune     acc = {ft_acc*100:.2f}%")

    assert pruned_nparams < base_nparams, "Pruning didn't reduce params!"
    print("\nOK")


if __name__ == "__main__":
    main()
