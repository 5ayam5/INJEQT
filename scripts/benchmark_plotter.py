from __future__ import annotations

from csv import DictReader
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmark_config import (
    FACTORY_TYPES,
    METRIC_LABELS,
    PLOT_COLUMNS,
    TDG_FACTORY_TYPES,
    benchmark_display_label,
    row_t_factory_type,
    rz_factories_for_t_factory_type,
)
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, MaxNLocator

PLOT_DPI = 200
NO_DATA_TEXT = "No valid data"
RZ_FRACTION_MODELS = ("TDG", "INJEQT")
VIRIDIS = plt.get_cmap("viridis")
INFERNO = plt.get_cmap("inferno")
MODEL_COLORS = {
    "TDG": INFERNO(0.25),
    "INJEQT": INFERNO(0.75),
    "INJEQT$^*$": INFERNO(0.75),
    "LatticeSurgery": VIRIDIS(0.2),
    "Transversal": VIRIDIS(0.8),
    "STAR": VIRIDIS(0.5),
}


def _adaptive_figsize(
    num_x_points: int, min_width: float = 5.0, num_plots: float | None = None
) -> tuple[float, float]:
    if num_x_points > 10:
        width = max(min_width, num_x_points * 0.5)
        height = width / 4.0
    else:
        height = max(min_width, num_x_points * 0.5) / 1.5
        width = height * 2.0
    return (width, height if num_plots is None else height * num_plots / 1.75)


def _baseline_label(t_factory_type: str) -> str:
    if t_factory_type == "STAR":
        return "best TDG mean"
    return "TDG"


def _set_y_axis_style(
    ax: Axes,
    values: list[float],
    *,
    use_log_scale: bool = False,
    clamp: tuple[float, float] | None = None,
) -> None:
    if not values:
        return

    if use_log_scale:
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        return

    min_value = min(values)
    max_value = max(values)
    span = max_value - min_value
    padding = 0.08 * span if span > 0 else max(0.05 * abs(max_value), 0.05)
    lower = min_value - padding
    upper = max_value + padding
    if clamp is not None:
        lower = max(lower, clamp[0])
        upper = min(upper, clamp[1])
    if lower == upper:
        upper = lower + 0.1

    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(MaxNLocator())


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(DictReader(f))


def _relative_improvement(
    baseline_value: float,
    candidate_value: float,
) -> float | None:
    if baseline_value <= 0 or candidate_value <= 0:
        return None
    return baseline_value / candidate_value


def _build_star_baseline_map(
    rows: list[dict[str, str]], metric: str
) -> dict[tuple[str, int], float]:
    per_benchmark_by_factory: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("model") != "TDG":
            continue
        t_factory_type = row_t_factory_type(row)
        if t_factory_type not in TDG_FACTORY_TYPES:
            continue
        raw_value = row.get(metric, "")
        if raw_value == "":
            continue
        per_benchmark_by_factory.setdefault(row["benchmark"], {}).setdefault(
            t_factory_type, []
        ).append(float(raw_value))

    baseline: dict[tuple[str, int], float] = {}
    for benchmark, values_by_factory in per_benchmark_by_factory.items():
        candidate_means = [
            sum(values) / len(values)
            for values in values_by_factory.values()
            if len(values) > 0
        ]
        if len(candidate_means) == 0:
            continue
        baseline[(benchmark, 0)] = min(candidate_means)
    return baseline


def _build_baseline_map(
    rows: list[dict[str, str]],
    metric: str,
    t_factory_type: str,
) -> dict[tuple[str, int], float]:
    if t_factory_type == "STAR":
        return _build_star_baseline_map(rows, metric)

    baseline: dict[tuple[str, int], float] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("model") != "TDG":
            continue
        if row.get("rz_factory") != "TFactory":
            continue
        if row_t_factory_type(row) != t_factory_type:
            continue
        if int(row.get("num_factories", 0) or 0) != 0:
            continue
        if row.get("policy", "") != "TDG":
            continue
        raw_value = row.get(metric, "")
        if raw_value == "":
            continue
        baseline[(row["benchmark"], int(row["trial"]))] = float(raw_value)
    return baseline


def _filter_candidate_rows(
    rows: list[dict[str, str]],
    model: str,
    rz_factory: str,
    t_factory_type: str,
):
    if rz_factory not in rz_factories_for_t_factory_type(t_factory_type):
        return
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("model") != model:
            continue
        if row.get("rz_factory") != rz_factory:
            continue
        if row_t_factory_type(row) != t_factory_type:
            continue
        yield row


def _collect_sweep_series(
    rows: list[dict[str, str]],
    metric: str,
    t_factory_type: str,
    candidate_rz_factory: str,
) -> dict[int, dict[str, list[float]]]:
    baseline = _build_baseline_map(rows, metric, t_factory_type)
    series: dict[int, dict[str, list[float]]] = {}
    for row in _filter_candidate_rows(
        rows, "INJEQT", candidate_rz_factory, t_factory_type
    ):
        if not row.get("policy", "").startswith("INJEQT_"):
            continue

        benchmark = row["benchmark"]
        trial = int(row["trial"])
        raw_value = row.get(metric, "")
        if raw_value == "":
            continue
        baseline_value = baseline.get((benchmark, trial), baseline.get((benchmark, 0)))
        if baseline_value is None:
            continue
        improvement = _relative_improvement(baseline_value, float(raw_value))
        if improvement is None:
            continue

        num_factories = int(row.get("num_factories", 0) or 0)
        series.setdefault(num_factories, {}).setdefault(benchmark, []).append(
            improvement
        )
    return series


def _compute_injeqt_star_series(
    sweep_series: dict[int, dict[str, list[float]]],
) -> tuple[dict[str, list[float]], dict[str, int]]:
    benchmark_names: set[str] = set()
    for benchmark_map in sweep_series.values():
        benchmark_names.update(benchmark_map.keys())

    best_series: dict[str, list[float]] = {}
    selected_num_factories: dict[str, int] = {}
    for benchmark in benchmark_names:
        best_num_factories: int | None = None
        best_mean = float("-inf")
        for num_factories, benchmark_map in sweep_series.items():
            values = benchmark_map.get(benchmark, [])
            if len(values) == 0:
                continue
            mean_value = sum(values) / len(values)
            if mean_value > best_mean:
                best_mean = mean_value
                best_num_factories = num_factories
        if best_num_factories is None:
            continue
        selected_num_factories[benchmark] = best_num_factories
        best_series[benchmark] = list(sweep_series[best_num_factories][benchmark])
    return best_series, selected_num_factories


def _save_no_data_plot(
    title: str, output_path: Path, figsize: tuple[float, float]
) -> None:
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.text(0.5, 0.5, NO_DATA_TEXT, ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI)
    plt.close()


def _save_current_plot(output_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI)
    plt.close()


def _should_use_log_scale(values: list[float]) -> bool:
    if not values:
        return False
    min_value = min(values)
    max_value = max(values)
    return min_value > 0 and max_value / min_value >= 10


def _plot_grouped_boxplot(
    benchmarks: list[str],
    benchmark_labels: list[str],
    series_data: dict[str, dict[str, list[float]]],
    title: str,
    output_path: Path,
    baseline_label: str,
) -> None:
    if len(benchmarks) == 0 or len(series_data) == 0:
        _save_no_data_plot(title, output_path, figsize=_adaptive_figsize(4))
        return

    labels = list(series_data.keys())
    x_positions = list(range(len(benchmarks)))
    avg_x_position = len(benchmarks)
    width = 0.75 / max(1, len(labels))
    legend_handles: list[Patch] = []

    all_plot_values = [
        v
        for factory_data in series_data.values()
        for benchmark_data in factory_data.values()
        for v in benchmark_data
    ]
    if not all_plot_values:
        _save_no_data_plot(title, output_path, figsize=_adaptive_figsize(4))
        return

    fig, ax = plt.subplots(figsize=_adaptive_figsize(len(benchmarks) + 1))
    for idx, label in enumerate(labels):
        benchmark_map = series_data[label]
        offset = (idx - (len(labels) - 1) / 2.0) * width
        positions: list[float] = []
        data: list[list[float]] = []
        all_values: list[float] = []
        for bench_index, benchmark in enumerate(benchmarks):
            values = benchmark_map.get(benchmark, [])
            if len(values) == 0:
                continue
            positions.append(bench_index + offset)
            data.append(values)
            all_values.extend(values)

        if len(data) == 0:
            continue

        color = MODEL_COLORS[label.split()[1]]
        boxplot = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            showfliers=False,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": color,
                "markeredgecolor": color,
                "markersize": 4,
            },
            medianprops={"color": color, "linewidth": 1.2},
            whiskerprops={"color": color},
            capprops={"color": color},
            boxprops={"edgecolor": color},
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        average_value = sum(all_values) / len(all_values)
        ax.bar(
            avg_x_position + offset,
            average_value,
            width=width * 0.9,
            color=color,
            edgecolor=color,
            alpha=0.8,
        )
        legend_handles.append(Patch(facecolor=color, edgecolor=color, label=label))

    ax.axhline(1.0, linewidth=1, linestyle="--")
    ax.set_xticks([*x_positions, avg_x_position])
    ax.set_xticklabels([*benchmark_labels, "Average"], rotation=35, ha="right")
    ax.set_xlabel("Benchmark")

    use_log = _should_use_log_scale(all_plot_values)
    if use_log:
        ax.set_yscale("log")
    _set_y_axis_style(ax, all_plot_values, use_log_scale=use_log)

    ax.set_ylabel(f"Improvement over\n{baseline_label} ($\\times$)")
    ax.set_title(title)
    ax.legend(handles=legend_handles)
    _save_current_plot(output_path)


def _plot_sweep_summary(
    sweep_series_by_factory: dict[str, dict[int, dict[str, list[float]]]],
    title: str,
    output_path: Path,
    baseline_label: str,
) -> None:
    all_num_factories = sorted(
        set().union(*(series.keys() for series in sweep_series_by_factory.values()))
    )
    all_num_factories = [n for n in all_num_factories if n % 2 == 1]
    if not all_num_factories:
        _save_no_data_plot(title, output_path, figsize=_adaptive_figsize(4))
        return

    active_factories: list[str] = []
    for rz_factory in sweep_series_by_factory:
        series = sweep_series_by_factory[rz_factory]
        if any(
            len(benchmark_map) > 0
            for num_factories in all_num_factories
            for benchmark_map in series.get(num_factories, {}).values()
        ):
            active_factories.append(rz_factory)

    if not active_factories:
        _save_no_data_plot(title, output_path, figsize=_adaptive_figsize(4))
        return

    x_positions = list(range(len(all_num_factories)))
    width = 0.75 / max(1, len(active_factories))
    all_plot_values: list[float] = []
    legend_handles: list[Patch] = []

    fig, ax = plt.subplots(figsize=_adaptive_figsize(len(all_num_factories)))
    for idx, rz_factory in enumerate(active_factories):
        offset = (idx - (len(active_factories) - 1) / 2.0) * width
        positions: list[float] = []
        violin_data: list[list[float]] = []

        for x_index, num_factories in enumerate(all_num_factories):
            benchmark_map = sweep_series_by_factory[rz_factory].get(num_factories, {})
            values = [
                v for per_benchmark in benchmark_map.values() for v in per_benchmark
            ]
            if len(values) == 0:
                continue
            positions.append(x_index + offset)
            violin_data.append(values)
            all_plot_values.extend(values)

        if len(violin_data) == 0:
            continue

        color = MODEL_COLORS[rz_factory]
        parts = ax.violinplot(
            violin_data,
            positions=positions,
            widths=width * 0.9,
            showmeans=True,
            showmedians=True,
        )
        for body in parts["bodies"]:  # type: ignore[index]
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.7)
        for partname in ("cbars", "cmins", "cmaxes", "cmedians", "cmeans"):
            if partname in parts:
                parts[partname].set_color(color)
                parts[partname].set_linewidth(1.3)

        legend_handles.append(Patch(facecolor=color, edgecolor=color, label=rz_factory))

    if not all_plot_values:
        _save_no_data_plot(title, output_path, figsize=_adaptive_figsize(4))
        return

    ax.axhline(1.0, linewidth=1, linestyle="--")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(num) for num in all_num_factories])
    ax.set_xlabel("#INJEQT factories")

    use_log = _should_use_log_scale(all_plot_values)
    if use_log:
        ax.set_yscale("log")
    _set_y_axis_style(ax, all_plot_values, use_log_scale=use_log)

    ax.set_ylabel(f"Improvement over\n{baseline_label} ($\\times$)")
    ax.set_title(title)
    ax.legend(handles=legend_handles)
    _save_current_plot(output_path)


def _collect_rz_injection_fractions_by_tfactory_and_model(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue

        t_factory_type = row_t_factory_type(row)
        model = row.get("model", "")
        rz_factory = row.get("rz_factory", "")
        if t_factory_type not in FACTORY_TYPES:
            continue

        if model not in RZ_FRACTION_MODELS:
            continue
        if model == "TDG":
            if t_factory_type not in TDG_FACTORY_TYPES:
                continue
        elif rz_factory not in rz_factories_for_t_factory_type(t_factory_type):
            continue

        rz_injection_error = row.get("rz_injection_error", "")
        total_error = row.get("total_error", "")
        if rz_injection_error == "" or total_error == "":
            continue

        rz_inj = float(rz_injection_error)
        tot_err = float(total_error)
        if tot_err <= 0:
            continue

        fraction = rz_inj / tot_err
        result.setdefault(t_factory_type, {}).setdefault(model, []).append(fraction)

    return result


def _plot_fraction_violin(
    data_by_tfactory: dict[str, dict[str, list[float]]],
    output_path: Path,
) -> None:
    if not data_by_tfactory:
        _save_no_data_plot(
            "RZ Injection Error Fraction",
            output_path,
            figsize=_adaptive_figsize(4),
        )
        return

    tfactory_types = sorted(data_by_tfactory.keys())
    violin_data_list: list[list[float]] = []
    positions_list: list[float] = []
    violin_colors: list[tuple[float, float, float, float]] = []
    all_values: list[float] = []

    offset = 0.5
    x_base = 0
    xtick_positions: list[float] = []
    xtick_labels: list[str] = []

    for tfactory in tfactory_types:
        xtick_positions.append(x_base)
        xtick_labels.append(tfactory)

        present_models = [
            m for m in RZ_FRACTION_MODELS if m in data_by_tfactory[tfactory]
        ]
        num_present = len(present_models)
        if num_present > 0:
            spacing = 2 * offset / (num_present + 1)
            for idx, model in enumerate(present_models):
                fractions = data_by_tfactory[tfactory][model]
                violin_data_list.append(fractions)
                all_values.extend(fractions)
                positions_list.append(x_base - offset + (idx + 1) * spacing)
                violin_colors.append(MODEL_COLORS[model])

        x_base += 1.5

    if not violin_data_list:
        _save_no_data_plot(
            "RZ Injection Error Fraction",
            output_path,
            figsize=_adaptive_figsize(4),
        )
        return

    fig, ax = plt.subplots(figsize=_adaptive_figsize(len(tfactory_types)))
    parts = ax.violinplot(
        violin_data_list,
        positions=positions_list,
        widths=0.3,
        showmeans=True,
        showmedians=True,
    )

    for i, pc in enumerate(parts["bodies"]):  # type: ignore[index]
        pc.set_facecolor(violin_colors[i])
        pc.set_edgecolor(violin_colors[i])
        pc.set_alpha(0.7)

    for partname in ("cbars", "cmins", "cmaxes", "cmedians", "cmeans"):
        if partname in parts:
            vp = parts[partname]
            vp.set_color(violin_colors)
            vp.set_linewidth(1.3)

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel("Fraction of Total Error\nfrom RZ Injection")
    ax.set_title("RZ Injection Error Fraction")
    _set_y_axis_style(ax, all_values, clamp=(0.0, 1.0))
    ax.grid(axis="y", alpha=0.3)

    legend_handles = [
        Patch(facecolor=MODEL_COLORS["TDG"], alpha=0.7, label="TDG"),
        Patch(facecolor=MODEL_COLORS["INJEQT"], alpha=0.7, label="INJEQT"),
    ]
    ax.legend(handles=legend_handles)
    _save_current_plot(output_path)


def _collect_selected_num_factories_by_tfactory_and_rzfactory(
    rows: list[dict[str, str]],
    metric: str,
) -> dict[str, dict[str, list[float]]]:
    selected_by_tfactory_and_rz: dict[str, dict[str, list[float]]] = {}
    for t_factory_type in TDG_FACTORY_TYPES:
        for rz_factory in rz_factories_for_t_factory_type(t_factory_type):
            sweep = _collect_sweep_series(
                rows=rows,
                metric=metric,
                t_factory_type=t_factory_type,
                candidate_rz_factory=rz_factory,
            )
            _, selected_num_factories = _compute_injeqt_star_series(sweep)
            selections = [float(v) for v in selected_num_factories.values()]
            if selections:
                selected_by_tfactory_and_rz.setdefault(t_factory_type, {})[
                    rz_factory
                ] = selections
    return selected_by_tfactory_and_rz


def _plot_selected_num_factories_violin(
    selected_by_tfactory_and_rz: dict[str, dict[str, list[float]]],
    metric: str,
    output_path: Path,
) -> None:
    if not selected_by_tfactory_and_rz:
        _save_no_data_plot(
            f"{METRIC_LABELS.get(metric, metric)}: Selected INJEQT* num_factories",
            output_path,
            figsize=_adaptive_figsize(4),
        )
        return

    labels = [t for t in TDG_FACTORY_TYPES if t in selected_by_tfactory_and_rz]
    if len(labels) == 0:
        _save_no_data_plot(
            f"{METRIC_LABELS.get(metric, metric)}: Selected INJEQT* num_factories",
            output_path,
            figsize=_adaptive_figsize(4),
        )
        return

    violin_data_list: list[list[float]] = []
    positions_list: list[float] = []
    violin_colors: list[tuple[float, float, float, float]] = []
    all_values: list[float] = []

    offset = 0.5
    x_base = 0
    xtick_positions: list[float] = []
    xtick_labels: list[str] = []

    for tfactory in labels:
        xtick_positions.append(x_base)
        xtick_labels.append(tfactory)

        rz_factories = rz_factories_for_t_factory_type(tfactory)
        present_rz = [
            rz for rz in rz_factories if rz in selected_by_tfactory_and_rz[tfactory]
        ]
        num_present = len(present_rz)
        if num_present > 0:
            spacing = 2 * offset / (num_present + 1)
            for idx, rz_factory in enumerate(present_rz):
                selections = selected_by_tfactory_and_rz[tfactory][rz_factory]
                violin_data_list.append(selections)
                all_values.extend(selections)
                positions_list.append(x_base - offset + (idx + 1) * spacing)
                violin_colors.append(MODEL_COLORS[rz_factory])

        x_base += 1.5

    if not violin_data_list:
        _save_no_data_plot(
            f"{METRIC_LABELS.get(metric, metric)}: Selected INJEQT* num_factories",
            output_path,
            figsize=_adaptive_figsize(4),
        )
        return

    fig, ax = plt.subplots(figsize=_adaptive_figsize(len(labels)))
    parts = ax.violinplot(
        violin_data_list,
        positions=positions_list,
        widths=0.3,
        showmeans=True,
        showmedians=True,
    )

    for i, pc in enumerate(parts["bodies"]):  # type: ignore[index]
        pc.set_facecolor(violin_colors[i])
        pc.set_edgecolor(violin_colors[i])
        pc.set_alpha(0.7)

    for partname in ("cbars", "cmins", "cmaxes", "cmedians", "cmeans"):
        if partname in parts:
            vp = parts[partname]
            vp.set_color(violin_colors)
            vp.set_linewidth(1.3)

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel("#Factories")
    ax.set_title(
        f"{METRIC_LABELS.get(metric, metric)}: Optimal INJEQT* factories across benchmarks"
    )
    _set_y_axis_style(ax, all_values)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)

    legend_handles = [
        Patch(facecolor=MODEL_COLORS[rz], alpha=0.7, label=rz)
        for rz in sorted(
            set(
                present_rz
                for d in selected_by_tfactory_and_rz.values()
                for present_rz in d.keys()
            )
        )
    ]
    if legend_handles:
        ax.legend(handles=legend_handles)

    _save_current_plot(output_path)


def _plot_combined_selected_num_factories_violin(
    metrics: list[str],
    selected_by_metric_and_tfactory_and_rz: dict[
        str, dict[str, dict[str, list[float]]]
    ],
    output_path: Path,
) -> None:
    num_metrics = len(metrics)
    if num_metrics == 0:
        return

    num_x_points = len(TDG_FACTORY_TYPES)

    fig, axes = plt.subplots(
        num_metrics,
        1,
        figsize=_adaptive_figsize(num_x_points, num_plots=num_metrics),
        sharex=True,
    )
    if num_metrics == 1:
        axes = [axes]

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        selected_by_tfactory_and_rz = selected_by_metric_and_tfactory_and_rz.get(
            metric, {}
        )

        if not selected_by_tfactory_and_rz:
            continue

        labels = [t for t in TDG_FACTORY_TYPES if t in selected_by_tfactory_and_rz]
        if len(labels) == 0:
            continue

        violin_data_list: list[list[float]] = []
        positions_list: list[float] = []
        violin_colors: list[tuple[float, float, float, float]] = []
        all_values: list[float] = []

        offset = 0.5
        x_base = 0
        xtick_positions: list[float] = []
        xtick_labels: list[str] = []

        for tfactory in labels:
            xtick_positions.append(x_base)
            xtick_labels.append(tfactory)

            rz_factories = rz_factories_for_t_factory_type(tfactory)
            present_rz = [
                rz for rz in rz_factories if rz in selected_by_tfactory_and_rz[tfactory]
            ]
            num_present = len(present_rz)
            if num_present > 0:
                spacing = 2 * offset / (num_present + 1)
                for idx, rz_factory in enumerate(present_rz):
                    selections = selected_by_tfactory_and_rz[tfactory][rz_factory]
                    violin_data_list.append(selections)
                    all_values.extend(selections)
                    positions_list.append(x_base - offset + (idx + 1) * spacing)
                    violin_colors.append(MODEL_COLORS[rz_factory])

            x_base += 1.5

        if not violin_data_list:
            continue

        parts = ax.violinplot(
            violin_data_list,
            positions=positions_list,
            widths=0.3,
            showmeans=True,
            showmedians=True,
        )

        for i, pc in enumerate(parts["bodies"]):  # type: ignore[index]
            pc.set_facecolor(violin_colors[i])
            pc.set_edgecolor(violin_colors[i])
            pc.set_alpha(0.7)

        for partname in ("cbars", "cmins", "cmaxes", "cmedians", "cmeans"):
            if partname in parts:
                vp = parts[partname]
                vp.set_color(violin_colors)
                vp.set_linewidth(1.3)

        ax.set_title(f"{METRIC_LABELS.get(metric, metric)}")
        _set_y_axis_style(ax, all_values)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=0.3)

        if ax_idx == num_metrics - 1:
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels(xtick_labels)

        if ax_idx == 0:
            legend_handles = [
                Patch(facecolor=MODEL_COLORS[rz], alpha=0.7, label=rz)
                for rz in sorted(
                    set(
                        present_rz
                        for d in selected_by_tfactory_and_rz.values()
                        for present_rz in d.keys()
                    )
                )
            ]
            if legend_handles:
                ax.legend(handles=legend_handles)

    fig.suptitle(
        "Optimal INJEQT* factories across benchmarks",
        fontsize=14,
        y=1.00,
    )
    fig.supylabel("Number of INJEQT Factories")
    plt.tight_layout()
    _save_current_plot(output_path)


def _plot_combined_boxplots(
    benchmarks: list[str],
    benchmark_labels: list[str],
    metrics: list[str],
    series_data_by_metric: dict[str, dict[str, dict[str, list[float]]]],
    t_factory_type: str,
    baseline_label: str,
    output_path: Path,
) -> None:
    num_metrics = len(metrics)
    if num_metrics == 0:
        return

    fig, axes = plt.subplots(
        num_metrics,
        1,
        figsize=_adaptive_figsize(len(benchmarks) + 1, num_plots=num_metrics),
        sharex=True,
    )
    if num_metrics == 1:
        axes = [axes]

    width = 0.8 / len(series_data_by_metric.get(metrics[0], {}))

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        series_data = series_data_by_metric.get(metric, {})

        all_plot_values = [
            v
            for factory_data in series_data.values()
            for benchmark_data in factory_data.values()
            for v in benchmark_data
        ]

        labels = sorted(series_data.keys())
        legend_handles: list[Patch] = []
        x_positions: list[float] = []
        avg_x_position = len(benchmarks) + 0.5

        for bench_index in range(len(benchmarks)):
            x_positions.append(bench_index)

        for idx, label in enumerate(labels):
            benchmark_map = series_data[label]
            offset = (idx - (len(labels) - 1) / 2.0) * width
            positions: list[float] = []
            data: list[list[float]] = []
            all_values: list[float] = []
            for bench_index, benchmark in enumerate(benchmarks):
                values = benchmark_map.get(benchmark, [])
                if len(values) == 0:
                    continue
                positions.append(bench_index + offset)
                data.append(values)
                all_values.extend(values)

            if len(data) == 0:
                continue

            color = MODEL_COLORS[label.split()[1]]
            boxplot = ax.boxplot(
                data,
                positions=positions,
                widths=width * 0.9,
                showfliers=False,
                patch_artist=True,
                showmeans=True,
                meanprops={
                    "marker": "o",
                    "markerfacecolor": color,
                    "markeredgecolor": color,
                    "markersize": 4,
                },
                medianprops={"color": color, "linewidth": 1.2},
                whiskerprops={"color": color},
                capprops={"color": color},
                boxprops={"edgecolor": color},
            )
            for patch in boxplot["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.8)

            if len(all_values) > 0:
                average_value = sum(all_values) / len(all_values)
                ax.bar(
                    avg_x_position + offset,
                    average_value,
                    width=width * 0.9,
                    color=color,
                    edgecolor=color,
                    alpha=0.8,
                )
            legend_handles.append(Patch(facecolor=color, edgecolor=color, label=label))

        ax.axhline(1.0, linewidth=1, linestyle="--")
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)}")

        use_log = _should_use_log_scale(all_plot_values) if all_plot_values else False
        if use_log:
            ax.set_yscale("log")
        _set_y_axis_style(ax, all_plot_values, use_log_scale=use_log)
        ax.grid(axis="y", alpha=0.3)

        if ax_idx == num_metrics - 1:
            ax.set_xticks([*x_positions, avg_x_position])
            ax.set_xticklabels([*benchmark_labels, "Average"], rotation=35, ha="right")
            ax.set_xlabel("Benchmark")

        if ax_idx == 0:
            ax.legend(handles=legend_handles)

    fig.suptitle(
        f"Improvement over {baseline_label} ({t_factory_type})",
        fontsize=14,
        y=1.00,
    )
    fig.supylabel(r"Improvement ($\times$)")
    plt.tight_layout()
    _save_current_plot(output_path)


def _plot_combined_sweeps(
    benchmarks: list[str],
    benchmark_labels: list[str],
    metrics: list[str],
    sweep_series_by_metric_and_factory: dict[
        str, dict[str, dict[int, dict[str, list[float]]]]
    ],
    t_factory_type: str,
    baseline_label: str,
    output_path: Path,
) -> None:
    num_metrics = len(metrics)
    if num_metrics == 0:
        return

    all_num_factories_unfiltered = set().union(
        *(
            series.keys()
            for factory_series in sweep_series_by_metric_and_factory.values()
            for series in factory_series.values()
        )
    )
    num_x_points = len([n for n in all_num_factories_unfiltered if n % 2 == 1])

    fig, axes = plt.subplots(
        num_metrics,
        1,
        figsize=_adaptive_figsize(
            num_x_points,
            num_plots=num_metrics,
        ),
        sharex=True,
    )
    if num_metrics == 1:
        axes = [axes]

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        sweep_series_by_factory = sweep_series_by_metric_and_factory.get(metric, {})

        all_plot_values: list[float] = []
        all_num_factories: list[int] = []

        for factory_series in sweep_series_by_factory.values():
            for num_factories in factory_series.keys():
                all_num_factories.append(num_factories)
                for benchmark_map in factory_series[num_factories].values():
                    all_plot_values.extend(benchmark_map)

        if not all_plot_values:
            continue

        all_num_factories = sorted(set(all_num_factories))
        all_num_factories = [n for n in all_num_factories if n % 2 == 1]
        legend_handles: list[Patch] = []
        width = 0.15

        for rz_factory, series in sweep_series_by_factory.items():
            offset = list(sweep_series_by_factory.keys()).index(rz_factory) * width - (
                (len(sweep_series_by_factory) - 1) * width / 2
            )
            x_positions: list[float] = []
            violin_data: list[list[float]] = []
            x_index = 0

            for num_factories in all_num_factories:
                benchmark_map = series.get(num_factories, {})
                values = [
                    v for per_benchmark in benchmark_map.values() for v in per_benchmark
                ]
                if len(values) == 0:
                    continue
                x_positions.append(x_index + offset)
                violin_data.append(values)
                x_index += 1

            if len(violin_data) == 0:
                continue

            color = MODEL_COLORS[rz_factory]
            parts = ax.violinplot(
                violin_data,
                positions=x_positions,
                widths=width * 0.9,
                showmeans=True,
                showmedians=True,
            )
            for body in parts["bodies"]:  # type: ignore[index]
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.7)
            for partname in ("cbars", "cmins", "cmaxes", "cmedians", "cmeans"):
                if partname in parts:
                    parts[partname].set_color(color)
                    parts[partname].set_linewidth(1.3)

            legend_handles.append(
                Patch(facecolor=color, edgecolor=color, label=rz_factory)
            )

        ax.axhline(1.0, linewidth=1, linestyle="--")
        x_ticks = list(range(len(all_num_factories)))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(num) for num in all_num_factories])
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)}")

        use_log = _should_use_log_scale(all_plot_values)
        if use_log:
            ax.set_yscale("log")
        _set_y_axis_style(ax, all_plot_values, use_log_scale=use_log)
        ax.grid(axis="y", alpha=0.3)

        if ax_idx == num_metrics - 1:
            ax.set_xlabel("Number of INJEQT factories")

        if ax_idx == 0:
            ax.legend(handles=legend_handles)

    fig.suptitle(
        f"INJEQT sweep over num_factories ({t_factory_type})",
        fontsize=14,
        y=1.00,
    )
    fig.supylabel(r"Improvement ($\times$)")
    plt.tight_layout()
    _save_current_plot(output_path)


def plot_from_csv(
    csv_path: Path, outputs_dir: Path, plot_metrics: list[str] | None = None
) -> None:
    if plot_metrics is None:
        plot_metrics = [
            "total_error",
            "space_time",
            "wall_clock_time",
            "num_physical_qubits",
        ]

    rows = load_rows(csv_path)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    for metric in PLOT_COLUMNS:
        for t_factory_type in FACTORY_TYPES:
            rz_factories = rz_factories_for_t_factory_type(t_factory_type)
            sweep_series_by_factory: dict[str, dict[int, dict[str, list[float]]]] = {}
            best_series_by_factory: dict[str, dict[str, list[float]]] = {}
            for rz_factory in rz_factories:
                sweep = _collect_sweep_series(
                    rows=rows,
                    metric=metric,
                    t_factory_type=t_factory_type,
                    candidate_rz_factory=rz_factory,
                )
                sweep_series_by_factory[rz_factory] = sweep
                best, _ = _compute_injeqt_star_series(sweep)
                best_series_by_factory[rz_factory] = best

            benchmark_order = sorted(
                set().union(*(bs.keys() for bs in best_series_by_factory.values()))
            )
            benchmark_labels = [
                benchmark_display_label(benchmark) for benchmark in benchmark_order
            ]
            series_data = {
                f"INJEQT$^*$ {rz}": best_series_by_factory.get(rz, {})
                for rz in rz_factories
            }
            baseline_label = _baseline_label(t_factory_type)

            _plot_grouped_boxplot(
                benchmarks=benchmark_order,
                benchmark_labels=benchmark_labels,
                series_data=series_data,
                title=(
                    f"{METRIC_LABELS[metric]}: Improvement over\n{baseline_label} ($\\times$) "
                    f"({t_factory_type})"
                ),
                output_path=(
                    outputs_dir
                    / f"boxplot_relative_{metric}_vs_tdg_{t_factory_type.lower()}.pdf"
                ),
                baseline_label=baseline_label,
            )

            _plot_sweep_summary(
                sweep_series_by_factory={
                    rz: sweep_series_by_factory.get(rz, {}) for rz in rz_factories
                },
                title=f"{METRIC_LABELS[metric]}: INJEQT sweep over num_factories ({t_factory_type})",
                output_path=outputs_dir
                / f"sweep_relative_{metric}_{t_factory_type.lower()}.pdf",
                baseline_label=baseline_label,
            )

        _plot_selected_num_factories_violin(
            selected_by_tfactory_and_rz=_collect_selected_num_factories_by_tfactory_and_rzfactory(
                rows, metric
            ),
            metric=metric,
            output_path=outputs_dir / f"selected_num_factories_{metric}.pdf",
        )

    for t_factory_type in FACTORY_TYPES:
        rz_factories = rz_factories_for_t_factory_type(t_factory_type)
        baseline_label = _baseline_label(t_factory_type)
        benchmark_order_combined: set[str] = set()
        series_data_by_metric: dict[str, dict[str, dict[str, list[float]]]] = {}
        sweep_series_by_metric_and_factory: dict[
            str, dict[str, dict[int, dict[str, list[float]]]]
        ] = {}

        for metric in plot_metrics:
            if metric not in PLOT_COLUMNS:
                continue

            best_series_by_factory: dict[str, dict[str, list[float]]] = {}
            sweep_series_by_factory: dict[str, dict[int, dict[str, list[float]]]] = {}

            for rz_factory in rz_factories:
                sweep = _collect_sweep_series(
                    rows=rows,
                    metric=metric,
                    t_factory_type=t_factory_type,
                    candidate_rz_factory=rz_factory,
                )
                sweep_series_by_factory[rz_factory] = sweep
                best, _ = _compute_injeqt_star_series(sweep)
                best_series_by_factory[rz_factory] = best
                benchmark_order_combined.update(best.keys())

            series_data_by_metric[metric] = {
                f"INJEQT$^*$ {rz}": best_series_by_factory.get(rz, {})
                for rz in rz_factories
            }
            sweep_series_by_metric_and_factory[metric] = sweep_series_by_factory

        if benchmark_order_combined:
            benchmark_order_list = sorted(benchmark_order_combined)
            benchmark_labels = [
                benchmark_display_label(benchmark) for benchmark in benchmark_order_list
            ]

            _plot_combined_boxplots(
                benchmarks=benchmark_order_list,
                benchmark_labels=benchmark_labels,
                metrics=plot_metrics,
                series_data_by_metric=series_data_by_metric,
                t_factory_type=t_factory_type,
                baseline_label=baseline_label,
                output_path=outputs_dir
                / f"boxplot_combined_{t_factory_type.lower()}.pdf",
            )

            _plot_combined_sweeps(
                benchmarks=benchmark_order_list,
                benchmark_labels=benchmark_labels,
                metrics=plot_metrics,
                sweep_series_by_metric_and_factory=sweep_series_by_metric_and_factory,
                t_factory_type=t_factory_type,
                baseline_label=baseline_label,
                output_path=outputs_dir
                / f"sweep_combined_{t_factory_type.lower()}.pdf",
            )

    selected_by_metric_and_tfactory_and_rz: dict[
        str, dict[str, dict[str, list[float]]]
    ] = {}
    for metric in plot_metrics:
        if metric not in PLOT_COLUMNS:
            continue
        selected_by_metric_and_tfactory_and_rz[metric] = (
            _collect_selected_num_factories_by_tfactory_and_rzfactory(rows, metric)
        )

    if selected_by_metric_and_tfactory_and_rz:
        _plot_combined_selected_num_factories_violin(
            metrics=plot_metrics,
            selected_by_metric_and_tfactory_and_rz=selected_by_metric_and_tfactory_and_rz,
            output_path=outputs_dir / "selected_num_factories_combined.pdf",
        )

    fractions_by_tfactory = _collect_rz_injection_fractions_by_tfactory_and_model(rows)
    if fractions_by_tfactory:
        _plot_fraction_violin(
            data_by_tfactory=fractions_by_tfactory,
            output_path=outputs_dir / "rz_injection_fraction.pdf",
        )
