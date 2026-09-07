"""Experiment 9 — pruning-method comparison across ResNet-18 / ViT /
MobileNetV2 on CIFAR-10.

Pipeline (per model):
 1. train the baseline model once and snapshot its weights;
 2. for each importance method (Taylor / Magnitude / Surrogate) restore
    the snapshot and prune it with the DependencyGraph pruner;
 3. measure accuracy right after pruning;
 4. fine-tune the pruned model;
 5. measure accuracy again.

A per-model summary table is printed at the end of the run.

Fixed parameter budget
----------------------
`global_pruning=False` (local, per-layer pruning) with the same
`--pruning-ratio` is used for every method. With local pruning the number
of channels removed from each dependency group is a function of the
group's root-layer size and the ratio only — it does NOT depend on the
importance scores. Therefore every method prunes exactly the same number
of parameters for a given model, and methods are compared fairly. The
pruned parameter count of the first method is remembered and verified
against the other methods.
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
from exp2_resnet18_cifar import make_cifar_resnet18, ignored_layers as resnet_ignored
from exp3_vit_cifar import make_vit, vit_pruner_kwargs
from exp8_taylor_mobilenetv2_cifar import (
    make_cifar_mobilenetv2, ignored_layers as mbv2_ignored,
)

import torch_pruning as tp


# ViT groups are Linear/LayerNorm-rooted, so magnitude/taylor need the
# extended target_types to score them (default is Conv/BN only).
VIT_TARGET_TYPES = (
    nn.modules.conv._ConvNd,
    nn.Linear,
    nn.modules.batchnorm._BatchNorm,
    nn.LayerNorm,
)


# ---------------------------------------------------------------------------
# model registry: builder, ignored layers, extra pruner kwargs, target types
# ---------------------------------------------------------------------------
def vit_ignored(model):
    return [model.head]


MODEL_SPECS = {
    "resnet18": dict(
        model_fn=make_cifar_resnet18,
        ignored_layers_fn=resnet_ignored,
        pruner_kwargs_fn=None,
        target_types=None,           # default (Conv/BN) covers ResNet groups
    ),
    "vit": dict(
        model_fn=make_vit,
        ignored_layers_fn=vit_ignored,
        pruner_kwargs_fn=vit_pruner_kwargs,
        target_types=VIT_TARGET_TYPES,
    ),
    "mobilenetv2": dict(
        model_fn=make_cifar_mobilenetv2,
        ignored_layers_fn=mbv2_ignored,
        pruner_kwargs_fn=None,
        target_types=None,
    ),
}


# ---------------------------------------------------------------------------
# importance registry
# ---------------------------------------------------------------------------
def build_importance(name: str, args, target_types):
    if name == "magnitude":
        return tp.importance.MagnitudeImportance(
            p=2, normalizer="mean",
            target_types=list(target_types) if target_types else None,
        )
    if name == "taylor":
        return tp.importance.TaylorImportance(
            normalizer="mean",
            target_types=list(target_types) if target_types else None,
        )
    if name == "surrogate":
        return tp.importance.SurrogateImportance(
            surrogate_epochs=args.surrogate_epochs,
            surrogate_lr=args.surrogate_lr,
            surrogate_batch_size=32,
            normalizer="mean",
            target_types=target_types,
        )
    raise ValueError(f"Unknown method: {name!r}")


def score_model(importance, model, pruner, calib_loader, criterion, device,
                method_name: str):
    """Method-specific scoring right before `pruner.step()`."""
    t0 = time.time()
    if isinstance(importance, tp.importance.SurrogateImportance):
        importance.fit(pruner, calib_loader, criterion, device=device)
        print(f"    [{method_name}] surrogate fit: {time.time()-t0:.1f}s, "
              f"{len(importance._imp)} groups scored")
    elif isinstance(importance, tp.importance.TaylorImportance):
        # TaylorImportance reads |grad * weight|, so accumulate gradients
        # over the calibration batch first.
        model.zero_grad()
        for x, y in calib_loader:
            x, y = x.to(device), y.to(device)
            criterion(model(x), y).backward()
        print(f"    [{method_name}] taylor grads: {time.time()-t0:.1f}s")
    # MagnitudeImportance is weight-only, nothing to prepare.


# ---------------------------------------------------------------------------
# one (model, method) run: prune -> acc -> finetune -> acc
# ---------------------------------------------------------------------------
def prune_and_finetune(method_name, base_state, spec, args,
                       train_loader, test_loader, calib_loader,
                       device, example_inputs, criterion,
                       expected_pruned_params) -> dict:
    model = spec["model_fn"]().to(device)
    model.load_state_dict(base_state)

    base_macs, base_nparams = count_stats(model, example_inputs)

    importance = build_importance(method_name, args, spec["target_types"])
    ignored = spec["ignored_layers_fn"](model)
    pruner_kwargs = dict(
        model=model,
        example_inputs=example_inputs,
        importance=importance,
        # LOCAL pruning: identical channel counts per group across methods.
        global_pruning=False,
        pruning_ratio=args.pruning_ratio,
        ignored_layers=ignored,
        round_to=args.round_to,
    )
    extra = spec["pruner_kwargs_fn"](model) if spec["pruner_kwargs_fn"] else None
    if extra:
        pruner_kwargs.update(extra)
    pruner = tp.pruner.MetaPruner(**pruner_kwargs)

    score_model(importance, model, pruner, calib_loader, criterion, device,
                method_name)
    pruner.step()

    pr_macs, pr_nparams = count_stats(model, example_inputs)
    # ---- fixed-budget check -------------------------------------------
    if expected_pruned_params is None:
        expected_pruned_params = pr_nparams
    elif pr_nparams != expected_pruned_params:
        print(f"    WARNING: pruned params {pr_nparams} != expected "
              f"{expected_pruned_params} — budget not fixed!")

    # 3. accuracy right after pruning
    pr_acc = evaluate(model, test_loader, device)
    print(f"    [{method_name}] pruned: params={pr_nparams/1e6:.2f}M "
          f"(−{(1-pr_nparams/base_nparams)*100:.1f}%), "
          f"MACs={pr_macs/1e9:.2f}G, acc={pr_acc*100:.2f}%")

    # 4. fine-tune
    if args.finetune_epochs > 0:
        opt = make_optimizer(args.optimizer, model,
                             args.finetune_lr, args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(args.finetune_epochs, 1))
        for epoch in range(args.finetune_epochs):
            t0 = time.time()
            loss, _ = train_one_epoch(model, train_loader, opt, criterion, device)
            sched.step()
            print(f"    [{method_name}] ft epoch {epoch+1}/{args.finetune_epochs}"
                  f"  loss={loss:.4f}  ({time.time()-t0:.1f}s)")

    # 5. accuracy after fine-tuning
    ft_acc = evaluate(model, test_loader, device)
    print(f"    [{method_name}] after finetune: acc={ft_acc*100:.2f}%")

    del model
    return dict(
        method=method_name,
        base_acc=None,          # filled by the caller
        pruned_acc=pr_acc,
        finetuned_acc=ft_acc,
        base_params=base_nparams,
        pruned_params=pr_nparams,
        base_macs=base_macs,
        pruned_macs=pr_macs,
    )


# ---------------------------------------------------------------------------
# full cycle for one model
# ---------------------------------------------------------------------------
def run_model(model_name, spec, args, train_loader, test_loader, calib_ds,
              device) -> dict:
    print(f"\n########## model: {model_name} ##########")
    example_inputs = torch.randn(1, 3, 32, 32, device=device)
    criterion = nn.CrossEntropyLoss()
    calib_loader = calibration_loader(calib_ds, args.calibration_samples,
                                      args.batch_size)

    # ---- 1. baseline training (once per model) ----
    model = spec["model_fn"]().to(device)
    opt = make_optimizer(args.optimizer, model,
                         args.lr, args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.epochs, 1))
    print(f"[baseline] training {args.epochs} epochs, "
          f"{args.optimizer} lr={args.lr}")
    for epoch in range(args.epochs):
        t0 = time.time()
        loss, tr_acc = train_one_epoch(model, train_loader, opt, criterion, device)
        sched.step()
        print(f"  epoch {epoch+1}/{args.epochs}  loss={loss:.4f}  "
              f"train_acc={tr_acc*100:.2f}%  ({time.time()-t0:.1f}s)")

    base_acc = evaluate(model, test_loader, device)
    base_macs, base_nparams = count_stats(model, example_inputs)
    print(f"[baseline] test_acc={base_acc*100:.2f}%  "
          f"params={base_nparams/1e6:.2f}M  MACs={base_macs/1e9:.2f}G")

    base_state = copy.deepcopy(model.state_dict())
    del model

    # ---- 2-5. every method against the same snapshot ----
    expected_pruned_params = None
    results = []
    for method in args.methods:
        print(f"  ---- method: {method} ----")
        r = prune_and_finetune(
            method, base_state, spec, args, train_loader, test_loader,
            calib_loader, device, example_inputs, criterion,
            expected_pruned_params,
        )
        r["base_acc"] = base_acc
        results.append(r)
        if expected_pruned_params is None:
            expected_pruned_params = r["pruned_params"]

    return dict(model=model_name, base_acc=base_acc,
                base_params=base_nparams, base_macs=base_macs,
                methods=results)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def print_report(all_results):
    print("\n" + "=" * 86)
    print("FINAL REPORT")
    print("=" * 86)
    for res in all_results:
        print(f"\n--- {res['model']}  "
              f"(baseline: acc={res['base_acc']*100:.2f}%, "
              f"params={res['base_params']/1e6:.2f}M, "
              f"MACs={res['base_macs']/1e9:.2f}G) ---")
        print(f"{'method':<12} {'params (M)':>11} {'Δparams %':>10} "
              f"{'MACs (G)':>9} {'acc pruned':>11} {'acc finetuned':>14} "
              f"{'Δacc ft':>9}")
        print("-" * 86)
        for r in res["methods"]:
            dp = (1 - r["pruned_params"] / r["base_params"]) * 100
            d_acc = (r["finetuned_acc"] - res["base_acc"]) * 100
            print(f"{r['method']:<12} {r['pruned_params']/1e6:>11.2f} "
                  f"{dp:>9.1f}%  {r['pruned_macs']/1e9:>9.2f} "
                  f"{r['pruned_acc']*100:>10.2f}%  "
                  f"{r['finetuned_acc']*100:>13.2f}%  "
                  f"{d_acc:>+8.2f}%")
    print("=" * 86)


# ---------------------------------------------------------------------------
def main():
    parser = make_common_parser(
        "exp9: Taylor/Magnitude/Surrogate x ResNet-18/ViT/MobileNetV2 / CIFAR-10")
    parser.add_argument("--models", nargs="+",
                        default=list(MODEL_SPECS.keys()),
                        choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--methods", nargs="+",
                        default=["taylor", "magnitude", "surrogate"],
                        choices=["taylor", "magnitude", "surrogate"])
    parser.add_argument("--round-to", type=int, default=1,
                        help="round pruned channel counts to a multiple of this")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")

    train_loader, test_loader, calib_ds = get_cifar10_loaders(
        args.data_dir, args.batch_size, args.num_workers)

    all_results = []
    for model_name in args.models:
        all_results.append(run_model(
            model_name, MODEL_SPECS[model_name], args,
            train_loader, test_loader, calib_ds, device))

    print_report(all_results)


if __name__ == "__main__":
    main()
