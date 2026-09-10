"""Collect accuracy-vs-parameter-pruning metrics for three CIFAR-10 models.

The experiment trains one baseline per model, restores that baseline for every
(method, pruning ratio) pair, and records accuracy before and after fine-tuning.
Metrics are saved as JSON and can be plotted independently with
``plot_exp10_pruning_curves.py``.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import torch
import torch.nn as nn

from _common import (
    calibration_loader,
    count_stats,
    evaluate,
    get_cifar10_loaders,
    make_common_parser,
    make_optimizer,
    set_seed,
    train_one_epoch,
)
from exp9_methods_x_models import MODEL_SPECS, prune_and_finetune


METHODS = ("taylor", "magnitude", "surrogate")


def save_metrics(path: Path, args, results: list[dict]) -> None:
    """Persist completed measurements, including partial long-running sweeps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "exp10_pruning_curves",
        "config": {
            "models": args.models,
            "methods": args.methods,
            "pruning_ratios": args.pruning_ratios,
            "epochs": args.epochs,
            "finetune_epochs": args.finetune_epochs,
            "seed": args.seed,
            "deterministic": args.deterministic,
        },
        "results": results,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def load_or_train_baseline(model_name, spec, args, train_loader, test_loader,
                           device, criterion, example_inputs):
    model = spec["model_fn"]().to(device)
    checkpoint_path = Path(args.checkpoint_dir) / f"{model_name}_seed{args.seed}.pt"
    train_config = {
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "deterministic": args.deterministic,
    }
    checkpoint = (
        torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint_path.exists() else None
    )
    if checkpoint is not None and checkpoint.get("train_config") == train_config:
        model.load_state_dict(checkpoint["state_dict"])
        print(f"[baseline] loaded checkpoint: {checkpoint_path}")
    else:
        optimizer = make_optimizer(
            args.optimizer, model, args.lr, args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1),
        )
        print(f"[baseline] training {args.epochs} epochs")
        for epoch in range(args.epochs):
            started = time.time()
            loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
            )
            scheduler.step()
            print(f"  epoch {epoch + 1}/{args.epochs}  loss={loss:.4f}  "
                  f"train_acc={train_acc * 100:.2f}%  "
                  f"({time.time() - started:.1f}s)")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(),
                    "train_config": train_config}, checkpoint_path)
        print(f"[baseline] saved checkpoint: {checkpoint_path}")

    accuracy = evaluate(model, test_loader, device)
    macs, parameters = count_stats(model, example_inputs)
    print(f"[baseline] acc={accuracy * 100:.2f}%  "
          f"params={parameters / 1e6:.2f}M  MACs={macs / 1e9:.2f}G")
    return model, float(accuracy), int(parameters), float(macs)


def run_model(model_name, spec, args, train_loader, test_loader, calib_ds,
              device, output_path, all_results) -> dict:
    print(f"\n########## model: {model_name} ##########")
    criterion = nn.CrossEntropyLoss()
    example_inputs = torch.randn(1, 3, 32, 32, device=device)
    calib = calibration_loader(
        calib_ds, args.calibration_samples, args.batch_size,
    )
    model, base_acc, base_params, base_macs = load_or_train_baseline(
        model_name, spec, args, train_loader, test_loader, device, criterion,
        example_inputs,
    )
    base_state = copy.deepcopy(model.state_dict())
    del model

    model_result = {
        "model": model_name,
        "baseline": {
            "accuracy": base_acc,
            "parameters": base_params,
            "macs": base_macs,
        },
        "measurements": [],
    }
    all_results.append(model_result)
    save_metrics(output_path, args, all_results)

    for ratio in args.pruning_ratios:
        for method in args.methods:
            print(f"  ---- ratio: {ratio:.3f}, method: {method} ----")
            args.pruning_ratio = ratio
            set_seed(args.seed, deterministic=args.deterministic)
            measurement = prune_and_finetune(
                method, base_state, spec, args, train_loader, test_loader,
                calib, device, example_inputs, criterion,
            )
            pruned_params = int(measurement["pruned_params"])
            model_result["measurements"].append({
                "method": method,
                "requested_pruning_ratio": ratio,
                "removed_parameters_percent":
                    (1.0 - pruned_params / base_params) * 100.0,
                "accuracy_before_finetuning":
                    float(measurement["pruned_acc"]),
                "accuracy_after_finetuning":
                    float(measurement["finetuned_acc"]),
                "parameters": pruned_params,
                "macs": float(measurement["pruned_macs"]),
            })
            save_metrics(output_path, args, all_results)
    return model_result


def validate_args(args) -> None:
    if not args.pruning_ratios:
        raise ValueError("At least one --pruning-ratios value is required")
    if any(ratio <= 0.0 or ratio >= 1.0 for ratio in args.pruning_ratios):
        raise ValueError("--pruning-ratios values must be between 0 and 1")
    if len(set(args.pruning_ratios)) != len(args.pruning_ratios):
        raise ValueError("--pruning-ratios values must be unique")


def main() -> None:
    parser = make_common_parser(
        "exp10: accuracy curves for pruning methods and CIFAR-10 models",
    )
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS),
                        choices=list(MODEL_SPECS))
    parser.add_argument("--methods", nargs="+", default=list(METHODS),
                        choices=METHODS)
    parser.add_argument("--pruning-ratios", nargs="+", type=float,
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--round-to", type=int, default=1)
    parser.add_argument("--max-pruning-ratio", type=float, default=0.9)
    parser.add_argument("--checkpoint-dir", default="./.checkpoints/exp10")
    parser.add_argument("--output", default="./results/exp10/metrics.json",
                        help="JSON file for metrics (updated after every run)")
    args = parser.parse_args()
    validate_args(args)

    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    print(f"Device: {device}")
    train_loader, test_loader, calib_ds = get_cifar10_loaders(
        args.data_dir, args.batch_size, args.num_workers,
    )

    output_path = Path(args.output)
    results = []
    for model_name in args.models:
        run_model(model_name, MODEL_SPECS[model_name], args, train_loader,
                  test_loader, calib_ds, device, output_path, results)
    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
