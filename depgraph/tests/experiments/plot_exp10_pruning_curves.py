"""Plot metrics produced by ``exp10_pruning_curves_metrics.py``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


MODEL_TITLES = {
    "resnet18": "ResNet-18",
    "vit": "ViT",
    "mobilenetv2": "MobileNetV2",
}
METHOD_LABELS = {
    "taylor": "Taylor",
    "taylor-mean": "Taylor",
    "magnitude": "L1",
    "magnitude-mean": "L1",
    "surrogate": "Graph",
    "surrogate2": "Graph",
}
METHOD_COLOR_INDEX = {
    "taylor": 0,
    "taylor-mean": 0,
    "magnitude": 1,
    "magnitude-mean": 1,
    "surrogate": 2,
    "surrogate2": 2,
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


def plot_models(
    results: list[dict], output_dir: Path, formats: list[str], dpi: int,
    selected_methods: set[str], output_name: str, title: str,
):
    # Full text width of the Springer LNCS proceedings template: 122 mm.
    width_inches = 122 / 25.4
    figure, axes = plt.subplots(
        2, 2, figsize=(width_inches, width_inches * 6.8 / 7.2),
        constrained_layout=True,
    )
    figure.suptitle(title)
    axes = axes.flatten()
    handles, labels = None, None
    for axis, result in zip(axes, results):
        model_name = result["model"]
        baseline = result["baseline"]
        measurements = result["measurements"]
        methods = list(dict.fromkeys(row["method"] for row in measurements))
        colors = sns.color_palette("colorblind", 3)
        palette = {method: colors[METHOD_COLOR_INDEX[method]] for method in methods}
        methods = [method for method in methods if method in selected_methods]

        axis.axhline(
            baseline["accuracy"] * 100.0,
            color="grey",
            linestyle="--",
            linewidth=0.6,
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
                    label=(METHOD_LABELS.get(method, method.title())
                           if metric == "accuracy_after_finetuning" else "_nolegend_"),
                    color=palette[method],
                    linestyle=line_style,
                    marker=marker,
                    linewidth=0.6,
                    markersize=1.2,
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
    method_legend = legend_axis.legend(
        handles,
        labels,
        loc="upper right",
        frameon=True,
    )
    legend_axis.add_artist(method_legend)
    legend_axis.legend(
        handles=[
            Line2D([], [], color="black", linestyle=line_style, marker=marker,
                   linewidth=0.6, markersize=1.2, label=state_label)
            for state_label, line_style, marker in LINE_STYLES.values()
        ],
        loc="lower right",
        frameon=True,
    )

    for extension in formats:
        destination = output_dir / f"{output_name}.{extension}"
        figure.savefig(destination, dpi=dpi)
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
            'font.size': 7,
            'figure.titlesize': 9,
            'axes.titlesize': 9,
            'axes.labelsize': 8,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
        },
    )
    results = [result for result in results if result["measurements"]]
    if results:
        mean_group_methods = {"taylor-mean", "magnitude-mean", "surrogate"}
        other_methods = {
            row["method"] for result in results for row in result["measurements"]
        } - mean_group_methods
        plot_models(
            results, output_dir, args.formats, args.dpi,
            mean_group_methods, "acc_vs_params_group",
            "Group-wise pruning",
        )
        plot_models(
            results, output_dir, args.formats, args.dpi,
            other_methods, "acc_vs_params_channel",
            "Channel-wise pruning",
        )


if __name__ == "__main__":
    main()
