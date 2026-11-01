"""Plot metrics produced by ``exp10_pruning_curves_metrics.py``."""
from __future__ import annotations

import argparse
import csv
import hashlib
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
    *, metric_styles=None, x_label="Removed parameters (%)",
    y_label="Test accuracy (%)", reference_label="Baseline",
    include_origin=True, x_limits=None, y_limits=None,
):
    metric_styles = LINE_STYLES if metric_styles is None else metric_styles
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
            label=reference_label,
            zorder=2,
        )
        for method in methods:
            rows = sorted(
                (row for row in measurements if row["method"] == method),
                key=lambda row: row["removed_parameters_percent"],
            )
            fitted_rows = sorted(
                (row for row in result.get("curve_measurements", measurements)
                 if row["method"] == method),
                key=lambda row: row["removed_parameters_percent"],
            )
            for metric, (state_label, line_style, marker) in metric_styles.items():
                x_values = ([0.0] if include_origin else []) + [
                    row["removed_parameters_percent"] for row in fitted_rows
                ]
                y_values = ([baseline["accuracy"] * 100.0] if include_origin else []) + [
                    row[metric] * 100.0 for row in fitted_rows
                ]
                axis.plot(
                    x_values,
                    y_values,
                    label=(METHOD_LABELS.get(method, method.title())
                           if metric == list(metric_styles)[-1] else "_nolegend_"),
                    color=palette[method],
                    linestyle=line_style,
                    marker=marker,
                    markevery=[] if "curve_measurements" in result else None,
                    linewidth=1,
                    markersize=1.2,
                )
                if "curve_measurements" in result:
                    axis.plot(
                        [row["removed_parameters_percent"] for row in rows],
                        [row[metric] * 100 for row in rows],
                        color=palette[method], linestyle="None", marker=marker,
                        markersize=1.2, label="_nolegend_",
                    )

        axis.set_title(MODEL_TITLES.get(model_name, model_name))
        if result.get("note"):
            axis.text(0.5, 0.5 if not measurements else 0.96, result["note"],
                      transform=axis.transAxes, ha="center", va="top")
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_xlim(left=0)
        if x_limits == "measured":
            axis.autoscale(enable=True, axis="x")
        elif x_limits is not None:
            axis.set_xlim(*x_limits)
        if y_limits is not None:
            axis.set_ylim(*y_limits)
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
            for state_label, line_style, marker in metric_styles.values()
        ],
        loc="lower right",
        frameon=True,
    )

    for extension in formats:
        destination = output_dir / f"{output_name}.{extension}"
        figure.savefig(destination, dpi=dpi)
        print(f"Saved: {destination}")
    plt.close(figure)


def content_variants(payloads, metric_paths, output_dir, formats, dpi, resume=False):
    """Compare measured tradeoffs without changing the existing visual design.

    All resampling is per seed, before averaging. Matched-budget comparisons
    use only the common measured support of every method and seed in a family.
    No interpolation from the unpruned baseline and no extrapolation are used.
    """
    import numpy as np
    from scipy.stats import t

    families = {
        "channel": ("Channel-wise", ("taylor", "magnitude", "surrogate2")),
        "group": ("Group-wise", ("taylor-mean", "magnitude-mean", "surrogate")),
    }
    models = [r["model"] for r in payloads[0]["results"]]
    states = list(LINE_STYLES)
    seeds = [p["config"]["seed"] for p in payloads]
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("The variants suite requires at least two distinct seeds")
    # Fail rather than silently mix experimental settings or incomplete sweeps.
    config = {k: v for k, v in payloads[0]["config"].items() if k != "seed"}
    data = {}
    for payload in payloads:
        if {k: v for k, v in payload["config"].items() if k != "seed"} != config:
            raise ValueError("Incompatible experiment configurations")
        if [r["model"] for r in payload["results"]] != models:
            raise ValueError("Different model lists across seeds")
        for result in payload["results"]:
            for method in config["methods"]:
                rows = [r for r in result["measurements"] if r["method"] == method]
                ratios = [r["requested_pruning_ratio"] for r in rows]
                if sorted(ratios) != sorted(config["pruning_ratios"]):
                    raise ValueError(f"Incomplete/duplicate sweep: {result['model']}, {method}")
                data[payload["config"]["seed"], result["model"], method] = (
                    result["baseline"], rows,
                )

    def budget(row, baseline, resource):
        if resource == "macs":
            return 100 * (1 - row["macs"] / baseline["macs"])
        return row["removed_parameters_percent"]

    def interpolate(seed, model, method, resource, state, xs):
        baseline, rows = data[seed, model, method]
        pairs = sorted((budget(r, baseline, resource), r[state] * 100) for r in rows)
        # Repeated measured budgets represent repeated observations, not a new x.
        unique_x = sorted(set(x for x, _ in pairs))
        ys = [mean(y for x, y in pairs if x == ux) for ux in unique_x]
        if min(xs) < unique_x[0] - 1e-9 or max(xs) > unique_x[-1] + 1e-9:
            raise ValueError("Attempted extrapolation")
        return np.interp(xs, unique_x, ys)

    def common_grid(model, methods, resource):
        supports, knots = [], []
        for seed in seeds:
            for method in methods:
                baseline, rows = data[seed, model, method]
                xs = [budget(r, baseline, resource) for r in rows]
                supports.append((min(xs), max(xs)))
                knots.extend(xs)
        lo, hi = max(x[0] for x in supports), min(x[1] for x in supports)
        if lo > hi:
            return np.array([])
        anchors = [x for x in [25, 50, 60, 70, 75, 80, 85, 90, 95, 98] if lo <= x <= hi]
        # The union of knots preserves every bend in the piecewise-linear
        # per-seed curves, including abrupt collapses at extreme compression.
        return np.unique([lo, hi] + [x for x in knots if lo <= x <= hi] + anchors)

    statistics_rows, seed_rows, catalog = [], [], []

    def record(name, model, method, state, xs, values):
        values = np.asarray(values, dtype=float)
        n = len(values)
        averages = values.mean(axis=0)
        deviations = values.std(axis=0, ddof=1) if n > 1 else np.zeros(len(xs))
        margins = t.ppf(.975, n - 1) * deviations / np.sqrt(n) if n > 1 else np.full(len(xs), np.nan)
        for j, x in enumerate(xs):
            statistics_rows.append(dict(
                variant=name, model=model, method=method, state=state,
                point_index=j, x=float(x), mean=float(averages[j]), std=float(deviations[j]),
                ci95_low=float(averages[j] - margins[j]),
                ci95_high=float(averages[j] + margins[j]), n=n,
                positive_seeds=int(np.sum(values[:, j] > 0)),
            ))
            for seed, value in zip(seeds, values[:, j]):
                seed_rows.append(dict(variant=name, model=model, method=method,
                                      state=state, point_index=j, x=float(x), seed=seed, value=float(value)))
        return averages

    def make_variant(family, number, slug, description, *, resource="parameters",
                     transform="accuracy", state_filter=None, matched=False,
                     reference=None, focus=None, threshold_max=None, only_seed=None):
        family_title, methods = families[family]
        name = f"{number:02d}_{slug}_{family}"
        if only_seed is not None:
            name += f"_seed{only_seed}"
        chosen_states = state_filter or states
        styles = {s: LINE_STYLES[s] for s in chosen_states}
        if transform == "recovery":
            chosen_states = [states[1]]
            styles = {states[1]: ("Fine-tuning recovery", "-", "s")}
        plotted = []
        for model in models:
            used_seeds = seeds if only_seed is None else [only_seed]
            baselines = [data[s, model, methods[0]][0]["accuracy"] * 100 for s in used_seeds]
            result = {"model": model, "baseline": {"accuracy": mean(baselines) / 100},
                      "measurements": []}
            if transform in {"loss", "advantage", "recovery", "feasible"}:
                result["baseline"]["accuracy"] = 0.0
            xs_common = (common_grid(model, methods, resource)
                         if matched and transform != "advantage" else None)
            if focus is not None:
                xs_common = xs_common[(xs_common >= focus[0]) & (xs_common <= focus[1])]
            missing = []
            for method in methods:
                if transform == "feasible":
                    xs = np.linspace(0, threshold_max, 41)
                elif matched:
                    xs = (common_grid(model, tuple(dict.fromkeys([method, reference])), resource)
                          if transform == "advantage" else xs_common)
                    if not len(xs):
                        missing.append(METHOD_LABELS[method])
                        continue
                else:
                    xs = np.asarray(sorted(config["pruning_ratios"]))
                curve = {}
                for state in chosen_states:
                    values, all_x = [], []
                    for seed in used_seeds:
                        baseline, rows = data[seed, model, method]
                        if transform == "feasible":
                            # An observed operating point must satisfy the loss
                            # constraint. Baseline (zero removal) is always feasible.
                            y = [max([0.] + [budget(r, baseline, resource) for r in rows
                                             if (baseline["accuracy"] - r[state]) * 100 <= loss + 1e-10])
                                 for loss in xs]
                            x = xs
                        elif matched:
                            x = xs
                            y = interpolate(seed, model, method, resource, state, xs)
                            if transform == "advantage":
                                y -= interpolate(seed, model, reference, resource, state, xs)
                            elif transform == "recovery":
                                y -= interpolate(seed, model, method, resource, states[0], xs)
                        else:
                            ordered = sorted(rows, key=lambda r: r["requested_pruning_ratio"])
                            x = np.array([budget(r, baseline, resource) for r in ordered])
                            y = np.array([r[state] * 100 for r in ordered])
                            if transform == "remaining":
                                x = 100 - x
                            elif transform == "compression":
                                x = 1 / (1 - x / 100)
                        if transform == "loss":
                            y = baseline["accuracy"] * 100 - np.asarray(y)
                        all_x.append(x)
                        values.append(y)
                    displayed_x = np.mean(all_x, axis=0)
                    if only_seed is None:
                        curve[state] = record(name, model, method, state, displayed_x, values)
                    else:
                        curve[state] = np.asarray(values[0])
                for j, x in enumerate(displayed_x):
                    row = {"method": method, "removed_parameters_percent": float(x)}
                    row.update({s: float(curve[s][j]) / 100 for s in chosen_states})
                    result["measurements"].append(row)
            if missing:
                result["note"] = ("No common measured budgets" if not result["measurements"]
                                  else f"{', '.join(missing)}: no shared budgets")
            plotted.append(result)
        xlabel = "Removed MACs (%)" if resource == "macs" else "Removed parameters (%)"
        ylabel, ref_label = "Test accuracy (%)", "Baseline"
        if transform == "loss":
            ylabel, ref_label = "Accuracy loss (pp)", "No loss"
        elif transform == "remaining":
            xlabel = "Remaining parameters (%)"
        elif transform == "compression":
            xlabel = "Parameter compression (x)"
        elif transform == "advantage":
            ylabel = f"Accuracy - {METHOD_LABELS[reference]} (pp)"
            ref_label = "Equal accuracy"
        elif transform == "recovery":
            ylabel, ref_label = "Fine-tuning gain (pp)", "No recovery"
        elif transform == "feasible":
            xlabel = "Allowed accuracy loss (pp)"
            ylabel, ref_label = "Achievable removal (%)", "Unpruned"
        if resume and all((output_dir / f"{name}.{fmt}").exists() for fmt in formats):
            print(f"Reused: {name}")
        else:
            plot_models(plotted, output_dir, formats, dpi, set(methods), name,
                        f"{family_title}: {description}" + (f"; seed {only_seed}" if only_seed is not None else ""),
                        metric_styles=styles, x_label=xlabel, y_label=ylabel,
                        reference_label=ref_label, include_origin=False,
                        x_limits="measured" if matched else None)
        catalog.append((name, description))

    for family, (_, methods) in families.items():
        after, before = [states[1]], [states[0]]
        make_variant(family, 1, "measured_both", "measured means")
        make_variant(family, 2, "measured_after", "after fine-tuning", state_filter=after)
        make_variant(family, 3, "measured_before", "before fine-tuning", state_filter=before)
        make_variant(family, 4, "loss_after", "loss after fine-tuning", transform="loss", state_filter=after)
        make_variant(family, 5, "loss_before", "loss before fine-tuning", transform="loss", state_filter=before)
        make_variant(family, 6, "remaining_after", "remaining parameters", transform="remaining", state_filter=after)
        make_variant(family, 7, "compression_after", "parameter compression", transform="compression", state_filter=after)
        make_variant(family, 8, "macs_both", "MAC reduction", resource="macs")
        make_variant(family, 9, "macs_after", "MACs after fine-tuning", resource="macs", state_filter=after)
        make_variant(family, 10, "matched_params", "equal parameter budgets", matched=True)
        make_variant(family, 11, "matched_macs", "equal MAC budgets", matched=True, resource="macs")
        for number, resource, ref in [(12, "parameters", methods[0]), (13, "parameters", methods[1]),
                                      (14, "macs", methods[0]), (15, "macs", methods[1])]:
            make_variant(family, number, f"advantage_{resource}_{ref}",
                         f"advantage over {METHOD_LABELS[ref]}", matched=True,
                         resource=resource, transform="advantage", reference=ref)
        make_variant(family, 16, "feasible_wide", "observed feasible pruning", transform="feasible", threshold_max=40)
        make_variant(family, 17, "feasible_tight", "small-loss operating points", transform="feasible", threshold_max=5)
        make_variant(family, 18, "recovery", "fine-tuning recovery", matched=True, transform="recovery")
        make_variant(family, 19, "high_pruning_after", "50-100% removal; after FT",
                     matched=True, state_filter=after, focus=(50, 100))
        for seed in seeds:
            make_variant(family, 20, "seed_after", "after fine-tuning", state_filter=after, only_seed=seed)
        for number, resource, ref in [(21, "parameters", methods[0]), (22, "parameters", methods[1]),
                                      (23, "macs", methods[0]), (24, "macs", methods[1])]:
            make_variant(family, number, f"advantage_after_{resource}_{ref}",
                         f"after FT vs {METHOD_LABELS[ref]}", matched=True, state_filter=after,
                         resource=resource, transform="advantage", reference=ref)
        for number, ref in [(25, methods[0]), (26, methods[1])]:
            make_variant(family, number, f"advantage_before_parameters_{ref}",
                         f"before FT vs {METHOD_LABELS[ref]}", matched=True, state_filter=before,
                         transform="advantage", reference=ref)

    for filename, rows in [("curve_statistics.csv", statistics_rows), ("per_seed_values.csv", seed_rows)]:
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "inputs": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in metric_paths],
        "seeds": seeds, "config": config, "figures": [name for name, _ in catalog],
        "formats": formats, "dpi": dpi,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_variants_readme(output_dir, formats, catalog, statistics_rows, seeds)


def write_variants_readme(output_dir, formats, catalog, statistics_rows, seeds):
    """Keep the generated figure directory self-contained and reproducible."""
    lines = [
        "# exp10: содержательные варианты графиков", "",
        f"Построено {len(catalog)} графиков; seeds: {', '.join(map(str, seeds))}.", "",
        "Изменён только `depgraph/tests/experiments/plot_exp10_pruning_curves.py`; "
        "исходные JSON и `results/exp10/plots` не менялись. Коммиты не создавались.", "",
        "## Что смотреть сначала", "",
        "- `02_measured_after_*`: точность после дообучения без сжатия шкалы низкими значениями до него.",
        "- `21_advantage_after_parameters_taylor_channel` и `22_advantage_after_parameters_magnitude_channel`: "
        "разница с Taylor и L1 при одинаковом проценте удаления. Выше нуля — преимущество соответствующего метода.",
        "- `23`/`24`: такая же проверка при одинаковом сокращении MACs. MACs — счётчик операций, не измеренная задержка.",
        "- `25`/`26`: преимущество до дообучения; на ViT картина существенно отличается от результатов после него.",
        "- `17_feasible_tight_*`: сколько параметров реально удалось удалить при малой допустимой потере точности.",
        "- `19_high_pruning_after_*`: крупнее виден режим сильного сжатия; полный диапазон есть в `02` и `10`.",
        "- `20_seed_after_*`: проверка, повторяется ли наблюдение в отдельных запусках.", "",
        "## Методика", "",
        "Оформление сохранено: сетка 2×2 (три модели и легенды), ширина 122 мм, отношение сторон, "
        "цвета Taylor/L1/Graph, тема, шрифты, толщина линий, маркеры и стили до/после дообучения. "
        "Во всех вариантах остаются линейные графики. Меняются метрики, сравнения, набор состояний и диапазон данных.", "",
        "1. В исходном режиме radius=0 усредняются только в точности совпавшие проценты удаления. "
        "Из-за разных структур это оставляло почти все точки отдельных seeds на одной ломаной. "
        "В `01`–`09` сначала сопоставляются одинаковые requested_pruning_ratio внутри метода, затем отдельно "
        "усредняются фактический бюджет и точность по seeds. Это описание измеренных рабочих точек, "
        "а не сравнение методов при строго одинаковых бюджетах.",
        "2. В `10`–`15`, `18`–`19`, `21`–`26` для каждого seed отдельно выполняется линейная интерполяция между измеренными "
        "бюджетами. Затем усредняются значения или парные разности. Для абсолютных кривых берётся пересечение "
        "диапазонов всех трёх методов и всех seeds. Для разностей — пересечение двух сравниваемых методов "
        "и всех seeds, отдельно для каждой пары. Если диапазоны не пересекаются, это явно подписано в панели; "
        "одна общая измеренная координата показывается одной точкой. Экстраполяции нет; "
        "точка непрореженной модели не используется как начало интерполяции. Общая сетка включает все измеренные "
        "узлы внутри пересечения, его границы и круглые контрольные бюджеты; изгибы интерполяции сохраняются. "
        "Ось X показывает общий измеренный диапазон каждой модели (в `19` дополнительно ограничен 50–100%). "
        "Поэтому group-wise сравнения могут иметь очень узкую область около экстремального сжатия. "
        "У ViT group-wise L1 всегда около 98.73% удаления параметров, а Graph не достигает этого бюджета: "
        "их прямое сравнение здесь невозможно. Graph и Taylor имеют общий диапазон примерно 91.4–95.6%.",
        "3. В `04`/`05` потеря считается относительно baseline того же seed в процентных пунктах: "
        "100 × (baseline accuracy − pruned accuracy). Меньше — лучше; отрицательные значения разрешены.",
        "4. В `16`/`17` для каждого seed и допустимой потери выбирается максимальное фактически измеренное "
        "удаление среди допустимых точек (либо 0 — исходная модель), затем берётся среднее. "
        "Между редкими точками возможны скачки; соединяющая линия не доказывает достижимость промежуточного бюджета. "
        "Критерий использует test accuracy и описывает эти данные, а не независимую настройку метода.",
        "5. `18` показывает after − before. Большое восстановление после дообучения само по себе "
        "не означает лучшего качества метода; это диагностический график.",
        "6. В новых вариантах не добавляется искусственный отрезок от baseline к первой измеренной точке. "
        "Горизонтальная линия обозначает baseline, нулевую потерю или равенство методов в зависимости от метрики. "
        "Шкалы линейные. Все три модели и все три метода семейства сохранены, включая проигрыши Graph.", "",
        "## Численная проверка", "",
        "Ниже — Graph после дообучения при равном бюджете параметров, channel-wise. "
        "Разности сначала считаются внутри seed. В квадратных скобках — 95% t-интервал среднего по seeds. "
        "Это описательные интервалы без поправки на множественные сравнения; интерполяция не заменяет новые запуски.", "",
        "| Модель | Удалено, % | Graph − Taylor, п.п. (95% CI) | Graph − L1, п.п. (95% CI) | Победы над Taylor / L1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODEL_TITLES:
        for x in [75, 85, 90, 95]:
            found = []
            for reference in ["taylor", "magnitude"]:
                found.append(next((r for r in statistics_rows
                                   if r["variant"] == f"{12 if reference == 'taylor' else 13:02d}_advantage_parameters_{reference}_channel"
                                   and r["model"] == model and r["method"] == "surrogate2"
                                   and r["state"] == "accuracy_after_finetuning" and abs(r["x"] - x) < 1e-8), None))
            if all(found):
                values = [f"{r['mean']:+.2f} [{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]" for r in found]
                wins = " / ".join(f"{r['positive_seeds']}/{r['n']}" for r in found)
                lines.append(f"| {MODEL_TITLES[model]} | {x} | {values[0]} | {values[1]} | {wins} |")
    lines += ["", "До дообучения (channel-wise, тот же бюджет параметров):", "",
              "| Модель | Удалено, % | Graph − Taylor, п.п. (95% CI) | Graph − L1, п.п. (95% CI) |",
              "|---|---:|---:|---:|"]
    for model, x in [("resnet18", 75), ("vit", 75), ("mobilenetv2", 75)]:
        values = []
        for number, ref in [(25, "taylor"), (26, "magnitude")]:
            row = next((r for r in statistics_rows
                        if r["variant"] == f"{number}_advantage_before_parameters_{ref}_channel"
                        and r["model"] == model and r["method"] == "surrogate2"
                        and abs(r["x"] - x) < 1e-8), None)
            if row:
                values.append(f"{row['mean']:+.2f} [{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}]")
        if len(values) == 2:
            lines.append(f"| {MODEL_TITLES[model]} | {x} | {values[0]} | {values[1]} |")
    lines += ["", "Преимущество зависит от модели и бюджета: оно хорошо видно для MobileNetV2 при сильном "
              "сжатии; у ResNet-18 преимущество над Taylor не всегда означает преимущество над L1. "
              "На ViT после дообучения в ряде диапазонов Graph проигрывает, хотя до дообучения имеет преимущество. "
              "Преимущество при равном числе параметров не переносится на равное сокращение MACs: "
              "в частности, на 75% и 90% сокращения MACs channel-wise Graph после дообучения проигрывает "
              "обоим конкурентам на всех трёх моделях. Универсального превосходства эти данные не подтверждают.", "",
              "`curve_statistics.csv`: для каждой точки графика среднее, SD, t-интервал, число seeds "
              "и число строго положительных значений (имеет смысл как число побед только для разностей). "
              "Единицы Y совпадают с подписью оси; точность хранится в процентах, разности — в п.п. "
              "`point_index` различает sweep-точки, которые могут иметь одинаковый фактический бюджет. "
              "`per_seed_values.csv`: значения до усреднения. Для `20` отдельные CSV не дублируются: "
              "это исходные измерения из JSON. `manifest.json`: конфигурация и SHA256 входных файлов.", "",
              "## Воспроизведение", "", "```sh",
              "MPLBACKEND=Agg MPLCONFIGDIR=/tmp/exp10-mpl python depgraph/tests/experiments/plot_exp10_pruning_curves.py \\",
              f"  --variants --metrics-dir results/exp10 --output-dir {output_dir} \\",
              f"  --formats {' '.join(formats)} --dpi 300", "```", "",
              "Без `--variants` сохранён прежний режим и имена файлов. В режиме вариантов `--group-radius` "
              "не используется: агрегация по sweep-точкам и сравнение при равном бюджете заданы явно.", "",
              "## Все варианты", "", "| График | Содержание |", "|---|---|"]
    for name, description in catalog:
        links = " · ".join(f"[{fmt}]({name}.{fmt})" for fmt in formats)
        lines.append(f"| `{name}` {links} | {description} |")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def lowess_measurements(rows: list[dict], fraction: float) -> list[dict]:
    """Local linear LOWESS at observed x, with tricube weights and no rejection.

    Unlike implementations that divide by a zero bandwidth at repeated x,
    a neighbourhood consisting of identical budgets returns their mean.
    Every input observation participates, including repeated measurements.
    """
    import numpy as np

    x = np.asarray([r["removed_parameters_percent"] for r in rows], dtype=float)
    y = np.asarray([[r[s] for s in LINE_STYLES] for r in rows], dtype=float)
    neighbours = min(len(rows), max(3, int(np.ceil(fraction * len(rows)))))
    fitted = []
    for target in np.unique(x):
        distances = np.abs(x - target)
        bandwidth = np.partition(distances, neighbours - 1)[neighbours - 1]
        if bandwidth == 0:
            prediction = y[x == target].mean(axis=0)
        else:
            weights = (1 - np.minimum(distances / bandwidth, 1) ** 3) ** 3
            selected = weights > 0
            design = np.column_stack([np.ones(selected.sum()), (x[selected] - target) / bandwidth])
            roots = np.sqrt(weights[selected])[:, None]
            prediction = np.linalg.lstsq(design * roots, y[selected] * roots, rcond=None)[0][0]
        row = {"method": rows[0]["method"], "removed_parameters_percent": float(target)}
        row.update({state: float(value) for state, value in zip(LINE_STYLES, prediction)})
        fitted.append(row)
    return fitted


def seed_linear_mean(seed_rows: list[tuple[int, list[dict]]]) -> tuple[list[dict], list[dict]]:
    """Interpolate within each seed, then average on their shared measured range."""
    import numpy as np

    curves = []
    for seed, rows in seed_rows:
        xs = sorted({r["removed_parameters_percent"] for r in rows})
        ys = np.asarray([[mean(r[state] for r in rows if r["removed_parameters_percent"] == x)
                          for x in xs] for state in LINE_STYLES])
        curves.append((seed, xs, ys))
    lower = max(xs[0] for _, xs, _ in curves)
    upper = min(xs[-1] for _, xs, _ in curves)
    if lower > upper:
        raise ValueError("Seeds have no shared measured range; cannot average without extrapolation")
    # The union of knots preserves every bend of the mean piecewise-linear curve.
    grid = sorted({lower, upper} | {x for _, xs, _ in curves for x in xs if lower <= x <= upper})
    interpolated = np.asarray([[np.interp(grid, xs, values) for values in ys]
                               for _, xs, ys in curves])
    averages = interpolated.mean(axis=0)
    rows, details = [], []
    for j, x in enumerate(grid):
        row = {"method": seed_rows[0][1][0]["method"], "removed_parameters_percent": float(x)}
        row.update({s: float(averages[i, j]) for i, s in enumerate(LINE_STYLES)})
        rows.append(row)
        for k, (seed, _, _) in enumerate(curves):
            for i, state in enumerate(LINE_STYLES):
                details.append(dict(seed=seed, method=row["method"], state=state,
                                    removed_parameters_percent=float(x), accuracy=float(interpolated[k, i, j])))
    return rows, details


def additional_curve_variants(payloads, metric_paths, output_dir, formats, dpi, radius, fractions):
    """Generate original-without-zero, raw+LOWESS, and mean linear-seed curves."""
    import copy

    families = {"channel": ("Channel-wise pruning", {"taylor", "magnitude", "surrogate2"}),
                "group": ("Group-wise pruning", {"taylor-mean", "magnitude-mean", "surrogate"})}
    seeds = [p["config"]["seed"] for p in payloads]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds in input files")
    config = {k: v for k, v in payloads[0]["config"].items() if k != "seed"}
    for payload in payloads:
        if {k: v for k, v in payload["config"].items() if k != "seed"} != config:
            raise ValueError("Incompatible experiment configurations")
    original = [r for r in average_results(payloads, radius=radius) if r["measurements"]]
    raw, linear = copy.deepcopy(original), copy.deepcopy(original)
    seed_details = []
    for result, averaged in zip(raw, linear):
        result["measurements"], averaged["measurements"] = [], []
        for method in config["methods"]:
            per_seed = []
            for payload in payloads:
                source = next(r for r in payload["results"] if r["model"] == result["model"])
                rows = [r for r in source["measurements"] if r["method"] == method]
                if not rows:
                    raise ValueError(f"Missing measurements: {result['model']}, {method}, seed {payload['config']['seed']}")
                per_seed.append((payload["config"]["seed"], rows))
                result["measurements"].extend(copy.deepcopy(rows))
            # Preserve the original common baseline point for the all-values LOWESS variant.
            result["measurements"].append(dict(
                method=method, removed_parameters_percent=0.,
                **{s: result["baseline"]["accuracy"] for s in LINE_STYLES},
            ))
            rows, details = seed_linear_mean(per_seed)
            averaged["measurements"].extend(rows)
            seed_details.extend(dict(model=result["model"], **row) for row in details)
    catalog, curve_values = [], []

    def save(name, results, description):
        for family, (title, methods) in families.items():
            filename = f"{name}_{family}"
            plot_models(results, output_dir, formats, dpi, methods, filename,
                        title, include_origin=False)
            catalog.append((filename, description))
        for result in results:
            for point_index, row in enumerate(result.get("curve_measurements", result["measurements"])):
                for state in LINE_STYLES:
                    curve_values.append(dict(variant=name, model=result["model"], method=row["method"],
                                             state=state, point_index=point_index,
                                             removed_parameters_percent=row["removed_parameters_percent"],
                                             accuracy=row[state]))

    save("01_original_no_zero", original, "Исходная агрегация, без добавленной точки x=0")
    for fraction in fractions:
        smoothed = copy.deepcopy(raw)
        for result in smoothed:
            result["curve_measurements"] = []
            for method in config["methods"]:
                rows = [r for r in result["measurements"] if r["method"] == method]
                result["curve_measurements"].extend(lowess_measurements(rows, fraction))
        save(f"02_lowess_{fraction:g}", smoothed, f"Все измерения + LOWESS, frac={fraction:g}")
    save("03_seed_linear_mean", linear, "Среднее линейно интерполированных кривых seeds")
    for filename, rows in [("curve_values.csv", curve_values), ("seed_interpolation.csv", seed_details)]:
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = dict(seeds=seeds, group_radius=radius, lowess_fractions=fractions,
                    figures=[name for name, _ in catalog], formats=formats, dpi=dpi,
                    inputs=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in metric_paths])
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = ["# Дополнительные варианты кривых exp10", "",
             "Сохранено оформление исходного скрипта: сетка 2×2, размеры, тема, цвета, шрифты, "
             "линии, маркеры и легенды. Ось X — удалённые параметры, Y — test accuracy. "
             "Оба состояния до/после дообучения и все три модели сохранены.", "",
             "1. `01_original_no_zero`: исходная агрегация `average_results` с заданным `--group-radius` "
             "и исходные отрезки; убрана только добавленная точка (0%, baseline accuracy). "
             "Нулевой тик оси и горизонтальная baseline остаются. Реальные измерения не удаляются.",
             "2. `02_lowess_*`: показаны маркеры **всех исходных измерений всех seeds**, даже при совпадающих x, "
             "плюс исходная общая точка baseline при x=0 для каждого метода. "
             "Линия — локальная линейная регрессия LOWESS с tricube-весами по ближайшим "
             "ceil(frac × N) наблюдениям (минимум 3). Робастного отбрасывания выбросов нет. "
             "При нулевой ширине окрестности берётся среднее всех значений в этой координате. "
             "Сглаженная линия не обязана проходить через отдельные маркеры, включая baseline. "
             "LOWESS вычисляется в измеренных координатах; между ними прямые отрезки. "
             "Показаны несколько одинаковых для всех методов уровней сглаживания. "
             "Сглаживание описывает наблюдения, не добавляет измеренных моделей; большие пробелы "
             "(особенно между baseline и первой точкой group-wise) остаются неподтверждёнными данными. "
             "Легенда сохраняет исходные обозначения маркеров и стилей состояний.",
             "3. `03_seed_linear_mean`: пункт «между похожими сидами» трактуется как сопоставление "
             "кривых одного метода/модели при **одинаковом фактическом удалении параметров**. "
             "Внутри каждого seed соседние измеренные x соединяются прямыми; значения на общей сетке "
             "усредняются с равным весом каждого seed. Повторяющиеся x внутри seed сначала усредняются. "
             "Для каждого метода берётся пересечение измеренных диапазонов всех seeds, "
             "без baseline, экстраполяции, отбора «удачных» seeds и переноса requested_pruning_ratio на ось X. "
             "Сетка — объединение всех узлов внутри пересечения: поэтому ломаная точно представляет среднее "
             "линейных интерполяций, а не приближение редкой сеткой. "
             "У разных методов могут быть разные диапазоны. При единственной общей координате показана одна точка.", "",
             "`curve_values.csv` содержит координаты линий в долях accuracy. "
             "`seed_interpolation.csv` — значения каждого seed до усреднения для варианта 3. "
             "`manifest.json` — seeds, параметры и SHA256 исходных JSON. Исходные графики и данные не менялись.", "",
             "```sh",
             "MPLBACKEND=Agg MPLCONFIGDIR=/tmp/exp10-mpl python depgraph/tests/experiments/plot_exp10_pruning_curves.py \\",
             f"  --curve-variants --metrics-dir {metric_paths[0].parent} --output-dir {output_dir} \\",
             f"  --formats {' '.join(formats)} --dpi {dpi} --group-radius {radius:g} \\",
             f"  --lowess-fracs {' '.join(f'{f:g}' for f in fractions)}", "```", "",
             "| График | Вариант |", "|---|---|"]
    for name, description in catalog:
        links = " · ".join(f"[{fmt}]({name}.{fmt})" for fmt in formats)
        lines.append(f"| {links} `{name}` | {description} |")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--variants", action="store_true",
                        help="Generate content-only comparisons and an audit README/CSV suite")
    modes.add_argument("--curve-variants", action="store_true",
                       help="Generate no-zero, LOWESS with all observations, and mean interpolated seed curves")
    parser.add_argument("--lowess-fracs", nargs="+", type=float, default=[0.25, 0.4, 0.6],
                        help="Neighbourhood fractions for --curve-variants")
    parser.add_argument("--resume-variants", action="store_true",
                        help="Reuse existing figures in an interrupted, otherwise identical variants run")
    args = parser.parse_args()
    if any(not 0 < fraction <= 1 for fraction in args.lowess_fracs):
        parser.error("--lowess-fracs values must be in (0, 1]")

    metric_paths = sorted(Path(args.metrics_dir).glob("*.json"))
    payloads = []
    for metric_path in metric_paths:
        with metric_path.open(encoding="utf-8") as file:
            payloads.append(json.load(file))
    if not payloads:
        raise FileNotFoundError(f"No JSON files found in {args.metrics_dir}")
    results = average_results(payloads, radius=args.group_radius)
    output_dir = Path(args.output_dir)
    if args.variants and output_dir == Path("./results/exp10/plots"):
        output_dir = Path("./results/exp10/content_variants")
    if args.curve_variants and output_dir == Path("./results/exp10/plots"):
        output_dir = Path("./results/exp10/curve_variants")
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
    if args.curve_variants:
        additional_curve_variants(payloads, metric_paths, output_dir, args.formats, args.dpi,
                                  args.group_radius, args.lowess_fracs)
        return
    if args.variants:
        content_variants(payloads, metric_paths, output_dir, args.formats, args.dpi,
                         resume=args.resume_variants)
        return
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
