"""Plot exp10 LOWESS curves without observation markers.

Use --lowess-frac (alias --frac) to change the smoothing fraction; default: 0.7.
The LOWESS calculation and visual design match the previous curve variants.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import seaborn as sns

from plot_exp10_pruning_curves import LINE_STYLES, average_results, lowess_measurements, plot_models


def smooth_results(payloads: list[dict], fraction: float = 0.7) -> list[dict]:
    """Fit all seeds together, retaining the original common baseline input."""
    if not 0 < fraction <= 1:
        raise ValueError("LOWESS fraction must be in (0, 1]")
    seeds = [p["config"]["seed"] for p in payloads]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds in input files")
    config = {k: v for k, v in payloads[0]["config"].items() if k != "seed"}
    for payload in payloads:
        if {k: v for k, v in payload["config"].items() if k != "seed"} != config:
            raise ValueError("Incompatible experiment configurations")
    results = [r for r in average_results(payloads) if r["measurements"]]
    for result in results:
        result["measurements"] = []
        for method in config["methods"]:
            rows = []
            for payload in payloads:
                source = next(r for r in payload["results"] if r["model"] == result["model"])
                measurements = [r for r in source["measurements"] if r["method"] == method]
                if not measurements:
                    raise ValueError(f"Missing measurements: {result['model']}/{method}, seed {payload['config']['seed']}")
                rows.extend(measurements)
            rows.append(dict(method=method, removed_parameters_percent=0.,
                             **{state: result["baseline"]["accuracy"] for state in LINE_STYLES}))
            fitted = lowess_measurements(rows, fraction)
            for row in fitted:
                if row["removed_parameters_percent"] == 0:
                    for state in LINE_STYLES:
                        row[state] = min(row[state], result["baseline"]["accuracy"])
            result["measurements"].extend(fitted)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/exp10"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp10/plots_smooth"))
    parser.add_argument("--lowess-frac", "--frac", type=float, default=0.7,
                        help="LOWESS neighbourhood fraction in (0, 1] (default: 0.7)")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg", "eps"],
                        default=["png", "pdf", "eps"])
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if not 0 < args.lowess_frac <= 1:
        parser.error("--lowess-frac must be in (0, 1]")
    paths = sorted(args.metrics_dir.glob("*.json"))
    if not paths:
        parser.error(f"No JSON metrics found in {args.metrics_dir}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    results = smooth_results(payloads, args.lowess_frac)
    if not results:
        parser.error("No measured curves found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(
        style="darkgrid", context="paper",
        rc={"figure.dpi": 120, "savefig.dpi": args.dpi,
            "font.size": 7, "figure.titlesize": 9, "axes.titlesize": 9,
            "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
            "legend.fontsize": 7},
    )
    # Only fitted lines are passed to the renderer. No raw or fitted markers.
    styles = {state: (label, linestyle, "None")
              for state, (label, linestyle, _) in LINE_STYLES.items()}
    for family, title, methods in [
        ("channel", "Channel-wise pruning", {"taylor", "magnitude", "surrogate2"}),
        ("group", "Group-wise pruning", {"taylor-mean", "magnitude-mean", "surrogate"}),
    ]:
        plot_models(results, args.output_dir, args.formats, args.dpi, methods,
                    f"acc_vs_params_{family}_smooth", title,
                    metric_styles=styles, include_origin=False)
    readme = f"""# LOWESS-кривые exp10 без исходных точек

Параметр LOWESS: `{args.lowess_frac:g}` (по умолчанию 0.7).
Используется тот же LOWESS, что в предыдущих вариантах: локальная линейная
регрессия с tricube-весами, без робастного отбрасывания наблюдений.
Все измерения всех seeds участвуют в расчёте. Совпадающие бюджеты обрабатываются
тем же алгоритмом; в окрестности нулевой ширины берётся среднее.
Как в предыдущем LOWESS-варианте, в расчёт для каждого метода добавляется
одна исходная точка (0%, средняя baseline accuracy).
LOWESS вычисляется в наблюдаемых координатах, между ними — прямые отрезки.

На рисунках показаны **только сглаженные линии** и горизонтальная baseline:
исходные точки и маркеры узлов сглаживания не отображаются.
Сохранены размеры, сетка 2×2, тема, цвета, шрифты, стили до/после дообучения.
Большие промежутки между измерениями не означают наличие измеренных моделей.
Исходные графики и JSON не изменены.

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/exp10-mpl python depgraph/tests/experiments/plot_exp10_pruning_curves_smooth.py \\
  --metrics-dir {args.metrics_dir} --output-dir {args.output_dir} \\
  --lowess-frac {args.lowess_frac:g} --formats {' '.join(args.formats)} --dpi {args.dpi}
```
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
