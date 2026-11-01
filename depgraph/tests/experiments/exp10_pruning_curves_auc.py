"""Export normalized AUCs of the original exp10 pruning curves as LaTeX.

These are accuracy-versus-pruning AUCs, not classification ROC-AUCs: exp10
does not contain false-positive/true-positive rates. Reuse the original plot's
aggregation, baseline point, and linear segments rather than the variants suite.

Example:
    python depgraph/tests/experiments/exp10_pruning_curves_auc.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from plot_exp10_pruning_curves import (
    LINE_STYLES,
    METHOD_LABELS,
    MODEL_TITLES,
    average_results,
)


FAMILIES = {
    "channel": ("Channel-wise", ("taylor", "magnitude", "surrogate2")),
    "group": ("Group-wise", ("taylor-mean", "magnitude-mean", "surrogate")),
}


def original_curve(result: dict, method: str, metric: str) -> list[tuple[float, float]]:
    """Return precisely the polyline used by the original plotting mode."""
    rows = sorted(
        (row for row in result["measurements"] if row["method"] == method),
        key=lambda row: row["removed_parameters_percent"],
    )
    if not rows:
        raise ValueError(f"No measurements for {result['model']}/{method}")
    points = [(0.0, result["baseline"]["accuracy"])] + [
        (row["removed_parameters_percent"], row[metric]) for row in rows
    ]
    if any(not math.isfinite(x) or not math.isfinite(y)
           or not 0 <= x <= 100 or not 0 <= y <= 1 for x, y in points):
        raise ValueError(f"Invalid curve values: {result['model']}/{method}/{metric}")
    return points


def integrate_curve(points: list[tuple[float, float]], upper: float) -> tuple[float, int]:
    """Integrate original linear segments on [0, upper], without extrapolation.

    Duplicate x values are retained: vertical segments have zero area, exactly
    as in the original graph. Their order determines adjacent linear segments.
    """
    if not math.isfinite(upper) or not 0 < upper <= points[-1][0]:
        raise ValueError(f"Integration endpoint {upper} is outside the curve")
    clipped = []
    for x, y in points:
        if x <= upper:
            clipped.append((x, y))
        else:
            left_x, left_y = clipped[-1]
            if left_x < upper:
                clipped.append((upper, left_y + (y - left_y) * (upper - left_x) / (x - left_x)))
            break
    area = sum((x2 - x1) * (y1 + y2) / 2
               for (x1, y1), (x2, y2) in zip(clipped, clipped[1:]))
    return area, len(clipped)


def latex_escape(value: str) -> str:
    replacements = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}",
                    "\\": r"\textbackslash{}", "~": r"\textasciitilde{}",
                    "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in value)


def latex_table(rows: list[dict], family: str,
                models: list[str], range_mode: str) -> str:
    family_title, methods = FAMILIES[family]
    states = list(LINE_STYLES)
    range_description = (
        "The integration range is shared by all methods within each model."
        if range_mode == "common" else
        "Each curve is integrated over its own plotted range; ranges may differ between methods."
    )
    values = {(row["model"], row["method"], row["state"]): row["normalized_auc"] for row in rows}
    best = {(model, state): max(values[model, method, state] for method in methods)
            for model in models for state in states}
    lines = [
        r"\begin{table}[H]",
        rf"\caption{{Normalized AUC of the original {family_title.lower()} accuracy--pruning curves "
        rf"(before/after fine-tuning). {range_description} Higher is better.}}",
        rf"\label{{results_exp10_auc_{family}}}",
        r"\centering",
        r"\begin{tabular}{|l|" + "c|" * len(models) + "}",
        r"\hline",
        r"\bfseries Method & " + " & ".join(latex_escape(MODEL_TITLES.get(m, m)) for m in models) + r" \\",
        r"\hline",
    ]
    for method in methods:
        cells = []
        for model in models:
            parts = []
            for state in states:
                value = values[model, method, state]
                part = rf"{value * 100:.1f}\%"
                if math.isclose(value, best[model, state], rel_tol=0, abs_tol=1e-12):
                    part = rf"\textbf{{{part}}}"
                parts.append(part)
            cells.append(" / ".join(parts))
        lines.append(latex_escape(METHOD_LABELS[method]) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/exp10"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp10/auc"))
    parser.add_argument("--group-radius", type=float, default=0.0,
                        help="Use the same grouping radius as the original plotting script")
    parser.add_argument("--range", dest="range_mode", choices=["common", "plotted"], default="common",
                        help="Common endpoint per model/family (default), or each curve's full plotted range")
    args = parser.parse_args()
    if not math.isfinite(args.group_radius) or args.group_radius < 0:
        parser.error("--group-radius must be finite and nonnegative")
    paths = sorted(args.metrics_dir.glob("*.json"))
    if not paths:
        parser.error(f"No JSON metrics found in {args.metrics_dir}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    results = [r for r in average_results(payloads, radius=args.group_radius) if r["measurements"]]
    if not results:
        parser.error("No measured curves found")
    models = [r["model"] for r in results]
    all_rows, tables = [], []
    for family, (_, methods) in FAMILIES.items():
        family_rows = []
        for metric in LINE_STYLES:
            rows = []
            for result in results:
                curves = {method: original_curve(result, method, metric) for method in methods}
                shared_upper = min(points[-1][0] for points in curves.values())
                for method, points in curves.items():
                    upper = shared_upper if args.range_mode == "common" else points[-1][0]
                    area, count = integrate_curve(points, upper)
                    rows.append(dict(
                        family=family, state=metric, model=result["model"], method=method,
                        x_start_percent=0.0, x_end_percent=upper,
                        area_accuracy_times_percent=area, normalized_auc=area / upper,
                        integration_points=count,
                    ))
            all_rows.extend(rows)
            family_rows.extend(rows)
        tables.append(latex_table(family_rows, family, models, args.range_mode))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = args.output_dir / "auc_tables.tex"
    tex_path.write_text("% Requires \\usepackage{float} for [H].\n\n" + "\n\n".join(tables) + "\n", encoding="utf-8")
    with (args.output_dir / "auc_values.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    readme = f"""# AUC исходных кривых exp10

`auc_tables.tex` содержит две LaTeX-таблицы: channel-wise и group-wise.
Столбцы — модели, строки — Taylor, L1 и Graph. В каждой ячейке значения
записаны через слеш: до / после дообучения.
Лучшее значение каждого столбца выделено жирным отдельно для каждого состояния.
Для `[H]` нужен `\\usepackage{{float}}`.
`auc_values.csv` содержит точные AUC, площади и границы интегрирования.

Это **AUC кривой accuracy vs removed parameters, а не ROC-AUC**:
в исходных данных нет FPR/TPR или оценок бинарного классификатора.
Нормированная AUC = ∫ accuracy(x) dx / (x_max − x_min).
Accuracy берётся в долях; в таблице AUC представлена процентами.
Таким образом, 80% — средняя точность 0.8 по диапазону удаления параметров,
а не вероятность правильного ранжирования положительного и отрицательного классов.

Расчёт точно повторяет ломаные исходного режима `plot_exp10_pruning_curves.py`:
`average_results`, радиус группировки {args.group_radius}, baseline при x=0,
сортировка по фактическому удалению параметров, линейные отрезки.
Используется метод трапеций; повторяющиеся x сохраняются и дают нулевую ширину.
Это площадь усреднённой исходным скриптом кривой, а не среднее площадей по seeds.
Недостатки исходной агрегации, описанные в `../content_variants/README.md`,
здесь намеренно сохраняются, чтобы считать именно исходные графики.

Режим диапазона: `{args.range_mode}`. По умолчанию `common`:
для каждой модели и семейства все методы интегрируются от 0 до минимального
из их максимальных измеренных x. Конечная точка при необходимости линейно
интерполируется; экстраполяции нет. Диапазоны разных моделей могут отличаться.
`--range plotted` использует весь исходный диапазон каждой кривой отдельно;
в этом режиме методы могут сравниваться на разных диапазонах.
Отрезок от baseline до первой измеренной точки берётся из исходного графика;
он не означает, что промежуточные модели действительно измерялись.

Запуск из корня проекта (прочитано JSON-файлов: {len(paths)}):

```sh
MPLCONFIGDIR=/tmp/exp10-mpl python depgraph/tests/experiments/exp10_pruning_curves_auc.py \\
  --metrics-dir {args.metrics_dir} --output-dir {args.output_dir} \\
  --group-radius {args.group_radius} --range {args.range_mode}
```
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Saved two LaTeX tables: {tex_path}")
    print(f"Saved {len(all_rows)} AUC values: {args.output_dir / 'auc_values.csv'}")


if __name__ == "__main__":
    main()
