"""Plot metrics produced by ``exp10_pruning_curves_metrics.py``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import seaborn as sns


MODEL_TITLES = {
    "resnet18": "ResNet-18",
    "vit": "ViT",
    "mobilenetv2": "MobileNetV2",
}
METHOD_LABELS = {
    "taylor": "Taylor exp.",
    "taylor-mean": "Taylor exp., mean",
    "magnitude": "L1",
    "magnitude-mean": "L1, mean",
    "surrogate": "Graph, group",
    "surrogate2": "Graph, channel",
}
LINE_STYLES = {
    "accuracy_before_finetuning": ("Before fine-tuning", "--", "o"),
    "accuracy_after_finetuning": ("After fine-tuning", "-", "s"),
}


def group_by_radius(rows: list[dict], radius: float) -> list[list[dict]]:
    """Cluster measurements whose removed-parameter values are within ``radius``."""
    groups: list[list[dict]] = []
    for row in sorted(rows, key=lambda row: row["removed_parameters_percent"]):
        if groups and (
            row["removed_parameters_percent"]
            - mean(r["removed_parameters_percent"] for r in groups[-1])
            <= radius
        ):
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


def average_results(payloads: list[dict], radius: float = 0.0) -> list[dict]:
    models = [result["model"] for result in payloads[0]["results"]]
    model_results = {model: {"baseline": [], "measurements": {}} for model in models}
    for payload in payloads:
        for result in payload["results"]:
            model = result["model"]
            model_results[model]["baseline"].append(result["baseline"]["accuracy"])
            for row in result["measurements"]:
                model_results[model]["measurements"].setdefault(
                    row["method"], []
                ).append(row)

    averaged = []
    for model in models:
        entry = model_results[model]
        reference = next(
            result for result in payloads[0]["results"] if result["model"] == model
        )
        measurements = []
        for method, rows in entry["measurements"].items():
            for group in group_by_radius(rows, radius):
                measurements.append({
                    "method": method,
                    "removed_parameters_percent": mean(
                        row["removed_parameters_percent"] for row in group
                    ),
                    "accuracy_before_finetuning": mean(
                        row["accuracy_before_finetuning"] for row in group
                    ),
                    "accuracy_after_finetuning": mean(
                        row["accuracy_after_finetuning"] for row in group
                    ),
                })
        averaged.append({
            "model": model,
            "baseline": {
                "accuracy": mean(entry["baseline"]),
                "parameters": reference["baseline"]["parameters"],
                "macs": reference["baseline"]["macs"],
            },
            "measurements": measurements,
        })
    return averaged


def plot_models(results: list[dict], output_dir: Path, formats: list[str], dpi: int):
    figure, axes = plt.subplots(
        2, 2, figsize=(7.2 * 2, 6.8 * 2), constrained_layout=True,
    )
    axes = axes.flatten()
    handles, labels = None, None
    for axis, result in zip(axes, results):
        model_name = result["model"]
        baseline = result["baseline"]
        measurements = result["measurements"]
        methods = list(dict.fromkeys(row["method"] for row in measurements))
        palette = dict(zip(methods, sns.color_palette("colorblind", len(methods))))

        axis.axhline(
            baseline["accuracy"] * 100.0,
            color="grey",
            linestyle="--",
            linewidth=1.8,
            label="Baseline",
            zorder=2,
        )
        for method in methods:
            rows = sorted(
                (row for row in measurements if row["method"] == method),
                key=lambda row: row["removed_parameters_percent"],
            )
            for metric, (state_label, line_style, marker) in LINE_STYLES.items():
                x_values = [0.0] + [
                    row["removed_parameters_percent"] for row in rows
                ]
                y_values = [baseline["accuracy"] * 100.0] + [
                    row[metric] * 100.0 for row in rows
                ]
                axis.plot(
                    x_values,
                    y_values,
                    label=f"{METHOD_LABELS.get(method, method.title())}, {state_label}",
                    color=palette[method],
                    linestyle=line_style,
                    marker=marker,
                    linewidth=1.8,
                    markersize=5,
                )

        axis.set_title(MODEL_TITLES.get(model_name, model_name))
        axis.set_xlabel("Removed parameters (%)")
        axis.set_ylabel("Test accuracy (%)")
        axis.set_xlim(left=0)
        axis.grid(True, which="major", alpha=0.75)
        sns.despine(ax=axis)
        if handles is None:
            handles, labels = axis.get_legend_handles_labels()

    legend_axis = axes[len(results)] if len(results) < len(axes) else axes[-1]
    if len(results) < len(axes):
        legend_axis.axis("off")
    legend_axis.legend(
        handles,
        labels,
        loc="upper right",
        frameon=True,
    )

    for extension in formats:
        destination = output_dir / f"acc_vs_params.{extension}"
        figure.savefig(destination, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {destination}")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot exp10 accuracy-vs-removed-parameters curves",
    )
    parser.add_argument("--metrics-dir", default="./results/exp10")
    parser.add_argument("--output-dir", default="./results/exp10/plots")
    parser.add_argument("--formats", nargs="+", default=["eps"],
                        choices=["png", "pdf", "svg", "eps"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--group-radius", type=float, default=0.0,
                        help="Radius for grouping removed_parameters_percent")
    args = parser.parse_args()

    metric_paths = sorted(Path(args.metrics_dir).glob("*.json"))
    payloads = []
    for metric_path in metric_paths:
        with metric_path.open(encoding="utf-8") as file:
            payloads.append(json.load(file))
    if not payloads:
        raise FileNotFoundError(f"No JSON files found in {args.metrics_dir}")
    results = average_results(payloads, radius=args.group_radius)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(
        style="darkgrid",
        context="paper",
        rc={
            "figure.dpi": 120,
            "savefig.dpi": args.dpi,
            'axes.titlesize': 21,
            'axes.labelsize': 20,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 21
        },
    )
    results = [result for result in results if result["measurements"]]
    if results:
        plot_models(results, output_dir, args.formats, args.dpi)


if __name__ == "__main__":
    main()
