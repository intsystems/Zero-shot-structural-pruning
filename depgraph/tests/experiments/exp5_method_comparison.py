"""Experiment 5 — side-by-side comparison of importance methods on ViT / CIFAR-10.

Trains the same small ViT ONCE, snapshots its weights, then for each
importance method (Magnitude, Taylor, Random, Surrogate) restores the
snapshot, prunes at the same target ratio, fine-tunes, and reports the
resulting accuracy. The point is to place `SurrogateImportance`
against standard baselines under an *identical* starting model.
"""
from __future__ import annotations

import copy
import time

import torch
import torch.nn as nn

from _common import (
    calibration_loader, count_stats, evaluate, get_cifar10_loaders,
    make_common_parser, make_optimizer, train_one_epoch,
)

import torch_pruning as tp
from exp3_vit_cifar import make_vit, vit_pruner_kwargs


VIT_TARGET_TYPES = (
    nn.modules.conv._ConvNd,
    nn.Linear,
    nn.modules.batchnorm._BatchNorm,
    nn.LayerNorm,
)


def build_importance(name: str, args):
    if name == "magnitude":
        return tp.importance.MagnitudeImportance(p=2, normalizer="mean",
                                                 target_types=list(VIT_TARGET_TYPES))
    if name == "taylor":
        return tp.importance.TaylorImportance(normalizer="mean",
                                              target_types=list(VIT_TARGET_TYPES))
    if name == "random":
        return tp.importance.RandomImportance()
    if name == "surrogate":
        return tp.importance.SurrogateImportance(
            surrogate_epochs=args.surrogate_epochs,
            surrogate_lr=args.surrogate_lr,
            surrogate_batch_size=32,
            normalizer="mean",
            target_types=VIT_TARGET_TYPES,
        )
    raise ValueError(f"Unknown method: {name!r}")


def prepare_importance(importance, model, calib_loader, criterion, device):
    """Method-specific setup right before `pruner.step()`."""
    if isinstance(importance, tp.importance.SurrogateImportance):
        # `pruner` is required for fit; assumes caller passes it in via prep_ctx
        return "needs_pruner"
    if isinstance(importance, tp.importance.TaylorImportance):
        model.zero_grad()
        for x, y in calib_loader:
            x, y = x.to(device), y.to(device)
            criterion(model(x), y).backward()
        return "ok"
    return "ok"


def prune_and_finetune(method_name, base_state, args, train_loader, test_loader,
                       calib_loader, device, example_inputs, criterion) -> dict:
    model = make_vit().to(device)
    model.load_state_dict(base_state)

    pre_acc = evaluate(model, test_loader, device)
    base_macs, base_nparams = count_stats(model, example_inputs)

    importance = build_importance(method_name, args)
    pruner = tp.pruner.MetaPruner(
        model, example_inputs,
        importance=importance,
        global_pruning=True,
        pruning_ratio=args.pruning_ratio,
        ignored_layers=[model.head],
        **vit_pruner_kwargs(model),
    )

    t0 = time.time()
    if isinstance(importance, tp.importance.SurrogateImportance):
        importance.fit(pruner, calib_loader, criterion, device=device)
        print(f"    [{method_name}] surrogate fit: {time.time()-t0:.1f}s, "
              f"{len(importance._imp)} groups scored")
    elif isinstance(importance, tp.importance.TaylorImportance):
        model.zero_grad()
        for x, y in calib_loader:
            x, y = x.to(device), y.to(device)
            criterion(model(x), y).backward()
        print(f"    [{method_name}] taylor grads: {time.time()-t0:.1f}s")

    pruner.step()
    pr_macs, pr_nparams = count_stats(model, example_inputs)
    pr_acc = evaluate(model, test_loader, device)
    print(f"    [{method_name}] after prune: acc={pr_acc*100:.2f}%  "
          f"params={pr_nparams/1e6:.2f}M "
          f"(−{(1-pr_nparams/base_nparams)*100:.1f}%)")

    if args.finetune_epochs > 0:
        opt = make_optimizer(args.optimizer, model.parameters(),
                             args.finetune_lr, args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(args.finetune_epochs, 1))
        for epoch in range(args.finetune_epochs):
            t0 = time.time()
            loss, _ = train_one_epoch(model, train_loader, opt, criterion, device)
            sched.step()
            print(f"    [{method_name}] ft epoch {epoch+1}/{args.finetune_epochs}"
                  f"  loss={loss:.4f}  ({time.time()-t0:.1f}s)")
        ft_acc = evaluate(model, test_loader, device)
    else:
        ft_acc = pr_acc

    return dict(
        method=method_name,
        pre_acc=pre_acc,
        pruned_acc=pr_acc,
        finetuned_acc=ft_acc,
        pruned_params=pr_nparams,
    )


def main():
    parser = make_common_parser("exp5: importance-method comparison on ViT / CIFAR-10")
    parser.add_argument("--methods", nargs="+",
                        default=["magnitude", "taylor", "random", "surrogate"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")

    train_loader, test_loader, calib_ds = get_cifar10_loaders(
        args.data_dir, args.batch_size, args.num_workers)
    calib_loader = calibration_loader(calib_ds, args.calibration_samples, args.batch_size)

    model = make_vit().to(device)
    example_inputs = torch.randn(1, 3, 32, 32, device=device)
    criterion = nn.CrossEntropyLoss()

    # ---- baseline train (ONCE) ----
    opt = make_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    print(f"[baseline] training {args.epochs} epochs")
    for epoch in range(args.epochs):
        t0 = time.time()
        loss, tr_acc = train_one_epoch(model, train_loader, opt, criterion, device)
        sched.step()
        print(f"  epoch {epoch+1}/{args.epochs}  loss={loss:.4f}  "
              f"train_acc={tr_acc*100:.2f}%  ({time.time()-t0:.1f}s)")

    base_acc = evaluate(model, test_loader, device)
    base_macs, base_nparams = count_stats(model, example_inputs)
    print(f"[baseline] test_acc={base_acc*100:.2f}%  params={base_nparams/1e6:.2f}M")

    base_state = copy.deepcopy(model.state_dict())
    del model

    # ---- run each method against the same starting checkpoint ----
    results = []
    for method in args.methods:
        print(f"\n---- method: {method} ----")
        r = prune_and_finetune(method, base_state, args, train_loader, test_loader,
                               calib_loader, device, example_inputs, criterion)
        results.append(r)

    # ---- summary table ----
    print("\n" + "=" * 78)
    print(f"{'method':<12} {'pre_acc':>10} {'pruned_acc':>12} {'finetuned':>12} {'params (M)':>12} {'Δparams %':>10}")
    print("-" * 78)
    print(f"{'BASELINE':<12} {base_acc*100:>9.2f}%  {'-':>12} {'-':>12} {base_nparams/1e6:>12.2f} {'-':>10}")
    for r in results:
        dp = (1 - r["pruned_params"] / base_nparams) * 100
        print(f"{r['method']:<12} {r['pre_acc']*100:>9.2f}%  "
              f"{r['pruned_acc']*100:>11.2f}%  {r['finetuned_acc']*100:>11.2f}%  "
              f"{r['pruned_params']/1e6:>12.2f} {dp:>9.1f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()
