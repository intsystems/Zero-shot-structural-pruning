"""Plot metrics produced by ``exp10_pruning_curves_metrics.py``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


MODEL_TITLES = {
    "resnet18": "ResNet-18",
    "vit": "ViT",
    "mobilenetv2": "MobileNetV2",
}
METHOD_LABELS = {
    "taylor": "Taylor expansion",
    "magnitude": "L1",
    "surrogate": "Graph",
}
LINE_STYLES = {
    "accuracy_before_finetuning": ("Before fine-tuning", "--", "o"),
    "accuracy_after_finetuning": ("After fine-tuning", "-", "s"),
}


def plot_model(result: dict, output_dir: Path, formats: list[str], dpi: int) -> None:
    model_name = result["model"]
    baseline = result["baseline"]
    measurements = result["measurements"]
    methods = list(dict.fromkeys(row["method"] for row in measurements))
    palette = dict(zip(methods, sns.color_palette("colorblind", len(methods))))

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for method in methods:
        rows = sorted(
            (row for row in measurements if row["method"] == method),
            key=lambda row: row["removed_parameters_percent"],
        )
        for metric, (state_label, line_style, marker) in LINE_STYLES.items():
            x_values = [0.0] + [row["removed_parameters_percent"] for row in rows]
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

    axis.set_title(f"Pruning Accuracy: {MODEL_TITLES.get(model_name, model_name)}")
    axis.set_xlabel("Removed parameters (%)")
    axis.set_ylabel("Test accuracy (%)")
    axis.set_xlim(left=0)
    axis.legend(fontsize=8, ncol=2, frameon=True)
    axis.grid(True, which="major", alpha=0.75)
    sns.despine(ax=axis)

    for extension in formats:
        destination = output_dir / f"{model_name}_accuracy_vs_parameters.{extension}"
        figure.savefig(destination, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {destination}")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot exp10 accuracy-vs-removed-parameters curves",
    )
    parser.add_argument("--metrics", default="./results/exp10/metrics.json")
    parser.add_argument("--output-dir", default="./results/exp10/plots")
    parser.add_argument("--formats", nargs="+", default=["eps"],
                        choices=["png", "pdf", "svg", "eps"])
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    with Path(args.metrics).open(encoding="utf-8") as file:
        payload = json.load(file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(
        style="darkgrid",
        context="paper",
        font_scale=1.15,
        rc={
            "figure.dpi": 120,
            "savefig.dpi": args.dpi,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "legend.title_fontsize": 9,
        },
    )
    for result in payload["results"]:
        if result["measurements"]:
            plot_model(result, output_dir, args.formats, args.dpi)


if __name__ == "__main__":
    main()
