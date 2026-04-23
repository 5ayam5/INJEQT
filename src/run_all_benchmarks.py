from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from csv import DictReader, DictWriter
from dataclasses import dataclass
from hashlib import blake2b
from math import ceil
from pathlib import Path
from pickle import load
from re import match, search
from typing import Any

import matplotlib
from ExecutionModels import (
    ColourCodeFactory,
    CultivationFactory,
    DistillationFactory,
    INJEQTExecutionModel,
    LatticeSurgerySurfaceCodeFactory,
    STARSurfaceCodeFactory,
    TDGExecutionModel,
    TransversalSurfaceCodeFactory,
)
from experiments import (
    NUM_LOGICAL_QUBITS_PER_MODULE,
    CompiledCirc,
    GrossCodeErrorModel,
    compute_synthesis_epsilon,
    count_noncliffords,
    evaluate_circuit,
    load_lookup_table,
)
from numpy.random import Generator, default_rng

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FACTORY_TYPES = (
    "Distillation",
    "Cultivation",
    "STAR",
)  # FIXME: add colour code back once implemented
TDG_FACTORY_TYPES = (
    "Distillation",
    "Cultivation",
)
FACTORY_TYPE_TO_RZ_FACTORIES = {
    "Distillation": ("LatticeSurgery", "Transversal"),
    "Cultivation": ("LatticeSurgery", "Transversal"),
    "STAR": ("STAR",),
}
STAR_BASELINE_NUM_FACTORIES = 1
STAR_BASELINE_POLICY = f"INJEQT_{STAR_BASELINE_NUM_FACTORIES}"
STATS_COLUMNS = [
    "#in_modules",
    "#inter_modules",
    "in_error",
    "inter_error",
    "rz_injection_error",
    "total_error",
    "num_gates",
    "time_in_module",
    "time_inter_module",
    "time_rz_injection",
    "active_time",
    "wall_clock_time",
    "num_physical_qubits",
    "space_time",
]
PLOT_COLUMNS = [
    "wall_clock_time",
    "active_time",
    "total_error",
    "num_physical_qubits",
    "space_time",
]
BASE_COLUMNS = [
    "benchmark",
    "num_program_bits",
    "model",
    "policy",
    "rz_factory",
    "t_factory_type",
    "num_factories",
    "trial",
    "seed",
    "synthesis_epsilon",
    "status",
    "error",
]

PLOT_DPI = 200
NO_DATA_TEXT = "No valid data"
SERIES_COLORS = (
    "#4472C4",
    "#ED7D31",
    "#70AD47",
    "#A5A5A5",
    "#5B9BD5",
    "#FFC000",
    "#264478",
    "#9E480E",
)
RZ_FACTORY_COLORS = {
    "LatticeSurgery": SERIES_COLORS[2],
    "Transversal": SERIES_COLORS[3],
    "STAR": SERIES_COLORS[4],
}
MODEL_COLORS = {
    "TDG": SERIES_COLORS[0],
    "INJEQT": SERIES_COLORS[1],
}
RZ_FRACTION_MODELS = ("TDG", "INJEQT")


@dataclass(frozen=True)
class RunConfig:
    model: str
    policy: str
    rz_factory: str
    t_factory_type: str
    stochastic: bool
    num_factories: int


@dataclass(frozen=True)
class RunJob:
    benchmark_name: str
    benchmark_path: str
    num_program_bits: int
    config: RunConfig
    trials: tuple[int, ...]
    seed_by_trial: tuple[int | None, ...]
    factory_distance: int
    synthesis_epsilon: float


_LOOKUP_TABLE: dict[int, int] | None = None


def _normalized_t_factory_type(t_factory_type: str, rz_factory: str) -> str:
    if t_factory_type:
        return t_factory_type
    if rz_factory == "STAR":
        return "STAR"
    return ""


def _row_t_factory_type(row: dict[str, str] | dict[str, Any]) -> str:
    return _normalized_t_factory_type(
        str(row.get("t_factory_type", "") or ""),
        str(row.get("rz_factory", "") or ""),
    )


def _rz_factories_for_t_factory_type(t_factory_type: str) -> tuple[str, ...]:
    return FACTORY_TYPE_TO_RZ_FACTORIES.get(t_factory_type, ())


def _init_worker(lookup: dict[int, int]) -> None:
    global _LOOKUP_TABLE
    _LOOKUP_TABLE = lookup


def spawn_child_rng(rng: Generator) -> Generator:
    return default_rng(int(rng.integers(0, 2**63 - 1)))


def load_benchmark(path: Path) -> CompiledCirc:
    with open(path, "rb") as f:
        return load(f)


def infer_num_program_bits(circuit: CompiledCirc) -> int:
    max_bit = -1
    for op in circuit.compiled_operations:
        for b in op[1]:
            max_bit = max(max_bit, int(b))
    return max_bit + 1


def run_seed(seed_root: int, cache_key: str) -> int:
    digest = blake2b(
        f"{seed_root}:{cache_key}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big") & ((1 << 63) - 1)


def parse_num_factories_sweep(spec: str) -> list[int]:
    tokens = [token.strip() for token in spec.split(",") if token.strip()]
    if len(tokens) == 0:
        raise ValueError("Empty --num-factories-sweep input.")

    values: list[int] = []
    for token in tokens:
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if start <= 0 or end <= 0:
                raise ValueError("All num-factories values must be positive integers.")
            step = 1 if end >= start else -1
            values.extend(list(range(start, end + step, step)))
            continue

        value = int(token)
        if value <= 0:
            raise ValueError("All num-factories values must be positive integers.")
        values.append(value)

    return list(dict.fromkeys(values))


def benchmark_display_label(benchmark_name: str, fallback_n: int | None = None) -> str:
    base = benchmark_name
    if base.endswith(".pkl"):
        base = base[: -len(".pkl")]
    base = base.replace(".qasm_after_commute", "")
    prefix = "official_compiled_circ_RzTranspiled_"
    if base.startswith(prefix):
        base = base[len(prefix) :]

    matched = search(r"([A-Za-z]+).*?_n(\d+)", base)
    if matched is None:
        matched = search(r"([A-Za-z]+).*?_(\d+)", base)
    if matched is not None:
        return f"{matched.group(1).lower()}_n{int(matched.group(2))}"

    if fallback_n is not None:
        name_match = match(r"([A-Za-z]+)", base)
        prefix_name = name_match.group(1).lower() if name_match is not None else base
        return f"{prefix_name}_n{fallback_n}"
    return base


def _compute_effective_distance(factory_distance: int, t_factory_type: str) -> int:
    if t_factory_type == "Cultivation":
        required_d = max(factory_distance, 2 * CultivationFactory.d_colour_code)
        if required_d % 2 == 0:
            required_d += 1
        return required_d
    return factory_distance


def build_execution_model(
    config: RunConfig,
    factory_distance: int,
    run_rng: Generator,
    num_modules: int,
    synthesis_epsilon: float,
):
    effective_distance = _compute_effective_distance(
        factory_distance, config.t_factory_type
    )

    if config.model == "TDG":
        if config.t_factory_type == "Distillation":
            factory_model = DistillationFactory(
                factory_distance,
                synthesis_epsilon=synthesis_epsilon,
            )
        elif config.t_factory_type == "Cultivation":
            factory_model = CultivationFactory(
                effective_distance,
                synthesis_epsilon=synthesis_epsilon,
                rng=spawn_child_rng(run_rng),
            )
        elif config.t_factory_type == "ColourCode":
            factory_model = ColourCodeFactory(
                factory_distance,
                synthesis_epsilon=synthesis_epsilon,
            )
        else:
            raise ValueError(f"Unknown T factory type: {config.t_factory_type}")
        return TDGExecutionModel(factory_model, num_modules)

    if config.rz_factory == "LatticeSurgery":
        factory_model = LatticeSurgerySurfaceCodeFactory(
            effective_distance,
            t_factory_type=config.t_factory_type,
            synthesis_epsilon=synthesis_epsilon,
            rng=spawn_child_rng(run_rng),
        )
    elif config.rz_factory == "Transversal":
        factory_model = TransversalSurfaceCodeFactory(
            effective_distance,
            t_factory_type=config.t_factory_type,
            synthesis_epsilon=synthesis_epsilon,
            rng=spawn_child_rng(run_rng),
        )
    elif config.rz_factory == "STAR":
        factory_model = STARSurfaceCodeFactory(
            effective_distance,
            rng=spawn_child_rng(run_rng),
        )
    else:
        raise ValueError(f"Unknown Rz factory: {config.rz_factory}")

    return INJEQTExecutionModel(
        factory_model,
        num_modules,
        num_factories=config.num_factories,
        rng=spawn_child_rng(run_rng),
    )


def run_job(job: RunJob) -> list[dict[str, Any]]:
    if _LOOKUP_TABLE is None:
        raise RuntimeError("Worker lookup table is not initialized.")

    circuit = load_benchmark(Path(job.benchmark_path))
    rows: list[dict[str, Any]] = []
    trial_failed = False
    trial_error = ""
    error_model = GrossCodeErrorModel()

    for index, trial in enumerate(job.trials):
        seed = job.seed_by_trial[index]
        row: dict[str, Any] = {
            "benchmark": job.benchmark_name,
            "num_program_bits": job.num_program_bits,
            "model": job.config.model,
            "policy": job.config.policy,
            "rz_factory": job.config.rz_factory,
            "t_factory_type": job.config.t_factory_type,
            "num_factories": job.config.num_factories,
            "trial": trial,
            "seed": seed if seed is not None else "",
            "synthesis_epsilon": job.synthesis_epsilon,
            "status": "ok",
            "error": "",
        }

        if trial_failed:
            row["status"] = "skipped"
            row["error"] = (
                f"Skipped because {index - 1} trial in this job failed: {trial_error}"
            )
            rows.append(row)
            continue

        run_rng = default_rng(seed) if seed is not None else default_rng()
        try:
            num_modules = ceil(job.num_program_bits / NUM_LOGICAL_QUBITS_PER_MODULE)
            execution_model = build_execution_model(
                config=job.config,
                factory_distance=job.factory_distance,
                run_rng=run_rng,
                num_modules=num_modules,
                synthesis_epsilon=job.synthesis_epsilon,
            )
            stats = evaluate_circuit(
                circuit=circuit,
                lookup=_LOOKUP_TABLE,
                num_program_bits=job.num_program_bits,
                error_model=error_model,
                execution_model=execution_model,
            )
            row.update(stats)
        except (
            NotImplementedError,
            TypeError,
            ValueError,
            AssertionError,
            RuntimeError,
        ) as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            trial_failed = True
            trial_error = row["error"]

        rows.append(row)

    job_label = (
        job.benchmark_name,
        job.config.policy,
        job.config.rz_factory,
        job.config.t_factory_type,
        job.config.num_factories,
    )
    print(f"Completed job {job_label} for trials {job.trials}")
    return rows


def all_configs(num_factories_sweep: list[int]) -> list[RunConfig]:
    configs: list[RunConfig] = []
    for t in TDG_FACTORY_TYPES:
        configs.append(
            RunConfig(
                model="TDG",
                policy="TDG",
                rz_factory="TFactory",
                t_factory_type=t,
                stochastic=t != "Distillation",
                num_factories=0,
            )
        )

    for t in FACTORY_TYPES:
        rz_factories = _rz_factories_for_t_factory_type(t)
        for num_factories in num_factories_sweep:
            for rz_factory in rz_factories:
                configs.append(
                    RunConfig(
                        model="INJEQT",
                        policy=f"INJEQT_{num_factories}",
                        rz_factory=rz_factory,
                        t_factory_type=t,
                        stochastic=True,
                        num_factories=num_factories,
                    )
                )

    return configs


def row_cache_key(row: dict[str, Any]) -> tuple[str, str, str, str, int, int, str]:
    num_factories = int(row.get("num_factories", 0) or 0)
    trial = int(row["trial"])
    t_factory_type = _row_t_factory_type(row)
    return (
        str(row["benchmark"]),
        str(row["model"]),
        str(row["rz_factory"]),
        t_factory_type,
        num_factories,
        trial,
        str(row.get("synthesis_epsilon", "") or ""),
    )


def read_existing_rows(
    csv_path: Path,
) -> dict[tuple[str, str, str, str, int, int, str], dict[str, str]]:
    if not csv_path.exists():
        return {}

    with open(csv_path, newline="") as f:
        rows = list(DictReader(f))

    latest: dict[tuple[str, str, str, str, int, int, str], dict[str, str]] = {}
    for row in rows:
        canonical = dict(row)
        canonical["t_factory_type"] = _row_t_factory_type(canonical)
        try:
            latest[row_cache_key(canonical)] = canonical
        except (KeyError, ValueError):
            continue
    return latest


def write_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*BASE_COLUMNS, *STATS_COLUMNS]
    with open(csv_path, "w", newline="") as f:
        writer = DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(payload)


def row_sort_key(row: dict[str, Any]) -> tuple[str, str, str, str, int, int, str]:
    t_factory_type = _row_t_factory_type(row)
    return (
        str(row["benchmark"]),
        str(row["model"]),
        str(row["rz_factory"]),
        t_factory_type,
        int(row.get("num_factories", 0) or 0),
        int(row["trial"]),
        str(row.get("synthesis_epsilon", "") or ""),
    )


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


def _build_baseline_map(
    rows: list[dict[str, str]],
    metric: str,
    t_factory_type: str,
) -> dict[tuple[str, int], float]:
    use_star_surrogate = t_factory_type == "STAR"
    baseline_model = "INJEQT" if use_star_surrogate else "TDG"
    baseline_rz_factory = "STAR" if use_star_surrogate else "TFactory"
    baseline_num_factories = STAR_BASELINE_NUM_FACTORIES if use_star_surrogate else 0
    baseline_policy = STAR_BASELINE_POLICY if use_star_surrogate else "TDG"

    baseline: dict[tuple[str, int], float] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("model") != baseline_model:
            continue
        if row.get("rz_factory") != baseline_rz_factory:
            continue
        if _row_t_factory_type(row) != t_factory_type:
            continue
        if int(row.get("num_factories", 0) or 0) != baseline_num_factories:
            continue
        if row.get("policy", "") != baseline_policy:
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
    if rz_factory not in _rz_factories_for_t_factory_type(t_factory_type):
        return
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("model") != model:
            continue
        if row.get("rz_factory") != rz_factory:
            continue
        if _row_t_factory_type(row) != t_factory_type:
            continue
        yield row


def _collect_relative_series(
    rows: list[dict[str, str]],
    metric: str,
    t_factory_type: str,
    candidate_rz_factory: str,
) -> dict[str, list[float]]:
    baseline = _build_baseline_map(rows, metric, t_factory_type)
    series: dict[str, list[float]] = {}
    for row in _filter_candidate_rows(
        rows, "INJEQT", candidate_rz_factory, t_factory_type
    ):
        benchmark = row["benchmark"]
        trial = int(row["trial"])
        raw_value = row.get(metric, "")
        if raw_value == "":
            continue
        candidate_value = float(raw_value)
        baseline_value = baseline.get((benchmark, trial), baseline.get((benchmark, 0)))
        if baseline_value is None:
            continue
        improvement = _relative_improvement(baseline_value, candidate_value)
        if improvement is None:
            continue
        series.setdefault(benchmark, []).append(improvement)
    return series


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
) -> None:
    if len(benchmarks) == 0 or len(series_data) == 0:
        _save_no_data_plot(title, output_path, figsize=(10, 4))
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
        _save_no_data_plot(title, output_path, figsize=(10, 4))
        return

    plt.figure(figsize=(max(12, len(benchmarks) * 0.6), 4))
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

        color = RZ_FACTORY_COLORS[label.split()[-1]]
        boxplot = plt.boxplot(
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
                "markersize": 5,
            },
            medianprops={"color": color, "linewidth": 1.2},
            whiskerprops={"color": color},
            capprops={"color": color},
            boxprops={"edgecolor": color},
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        average_value = sum(all_values) / len(all_values)
        plt.bar(
            avg_x_position + offset,
            average_value,
            width=width * 0.9,
            color=color,
            edgecolor=color,
            alpha=0.85,
        )
        legend_handles.append(Patch(facecolor=color, edgecolor=color, label=label))

    plt.axhline(1.0, linewidth=1, linestyle="--")
    plt.xticks(
        [*x_positions, avg_x_position],
        [*benchmark_labels, "Average"],
        rotation=35,
        ha="right",
    )
    plt.xlabel("Benchmark")

    if _should_use_log_scale(all_plot_values):
        plt.yscale("log")

    plt.ylabel(r"Improvement over TDG ($\times$)")
    plt.title(title)
    plt.legend(handles=legend_handles)
    _save_current_plot(output_path)


def _plot_sweep_summary(
    sweep_series_by_factory: dict[str, dict[int, dict[str, list[float]]]],
    title: str,
    output_path: Path,
) -> None:
    all_num_factories = sorted(
        set().union(*(series.keys() for series in sweep_series_by_factory.values()))
    )
    if not all_num_factories:
        _save_no_data_plot(title, output_path, figsize=(6, 4))
        return

    active_factories: list[str] = []
    for rz_factory in sweep_series_by_factory:
        series = sweep_series_by_factory[rz_factory]
        if any(
            len([v for per_benchmark in benchmark_map.values() for v in per_benchmark])
            > 0
            for benchmark_map in series.values()
        ):
            active_factories.append(rz_factory)

    if not active_factories:
        _save_no_data_plot(title, output_path, figsize=(6, 4))
        return

    x_positions = list(range(len(all_num_factories)))
    width = 0.75 / max(1, len(active_factories))
    all_plot_values: list[float] = []
    legend_handles: list[Patch] = []

    plt.figure(figsize=(max(6, len(all_num_factories) * 0.8), 4))
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

        color = RZ_FACTORY_COLORS.get(rz_factory, RZ_FACTORY_COLORS["STAR"])
        parts = plt.violinplot(
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
                parts[partname].set_linewidth(1.5)

        legend_handles.append(Patch(facecolor=color, edgecolor=color, label=rz_factory))

    if not all_plot_values:
        _save_no_data_plot(title, output_path, figsize=(6, 4))
        return

    plt.axhline(1.0, linewidth=1, linestyle="--")
    plt.xticks(x_positions, [str(num) for num in all_num_factories])
    plt.xlabel("Number of INJEQT factories")

    if _should_use_log_scale(all_plot_values):
        plt.yscale("log")

    plt.ylabel("Improvement over TDG (x)")
    plt.title(title)
    plt.legend(handles=legend_handles)
    _save_current_plot(output_path)


def _collect_rz_injection_fractions_by_tfactory_and_model(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, list[float]]]:
    """
    Collect RZ injection fractions grouped by factory type and model.
    Returns dict[t_factory_type][model] = [fractions].
    """
    result: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue

        t_factory_type = _row_t_factory_type(row)
        model = row.get("model", "")
        rz_factory = row.get("rz_factory", "")
        if t_factory_type not in FACTORY_TYPES:
            continue

        if model not in RZ_FRACTION_MODELS:
            continue
        if model == "TDG":
            if t_factory_type not in TDG_FACTORY_TYPES:
                continue
        elif rz_factory not in _rz_factories_for_t_factory_type(t_factory_type):
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
    """
    Plot RZ injection fractions as violin plot with T factory types on x-axis
    and TDG/INJEQT violins side by side for each type.
    """
    if not data_by_tfactory:
        _save_no_data_plot("RZ Injection Error Fraction", output_path, figsize=(8, 4))
        return

    tfactory_types = sorted(data_by_tfactory.keys())

    violin_data_list: list[list[float]] = []
    positions_list: list[float] = []
    violin_colors: list[str] = []

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
                positions_list.append(x_base - offset + (idx + 1) * spacing)
                violin_colors.append(MODEL_COLORS[model])

        x_base += 1.5

    if not violin_data_list:
        _save_no_data_plot("RZ Injection Error Fraction", output_path, figsize=(8, 4))
        return

    _, ax = plt.subplots(figsize=(max(6, len(tfactory_types) * 1.5), 3))

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
            vp.set_linewidth(1.5)

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel("Fraction of Total Error from RZ Injection")
    ax.set_title("RZ Injection Error Fraction")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    legend_handles = [
        Patch(facecolor=MODEL_COLORS["TDG"], alpha=0.7, label="TDG"),
        Patch(facecolor=MODEL_COLORS["INJEQT"], alpha=0.7, label="INJEQT"),
    ]
    ax.legend(handles=legend_handles)

    _save_current_plot(output_path)


def plot_from_csv(csv_path: Path, outputs_dir: Path) -> None:
    rows = load_rows(csv_path)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    for metric in PLOT_COLUMNS:
        for t_factory_type in FACTORY_TYPES:
            rz_factories = _rz_factories_for_t_factory_type(t_factory_type)
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

            _plot_grouped_boxplot(
                benchmarks=benchmark_order,
                benchmark_labels=benchmark_labels,
                series_data=series_data,
                title=(
                    rf"{metric}: Improvement over TDG ($\times$) ({t_factory_type})"
                ),
                output_path=(
                    outputs_dir
                    / f"boxplot_relative_{metric}_vs_tdg_{t_factory_type.lower()}.pdf"
                ),
            )

            _plot_sweep_summary(
                sweep_series_by_factory={
                    rz: sweep_series_by_factory.get(rz, {}) for rz in rz_factories
                },
                title=f"{metric}: INJEQT sweep over num_factories ({t_factory_type})",
                output_path=outputs_dir
                / f"sweep_relative_{metric}_{t_factory_type.lower()}.pdf",
            )

    fractions_by_tfactory = _collect_rz_injection_fractions_by_tfactory_and_model(rows)

    if fractions_by_tfactory:
        _plot_fraction_violin(
            data_by_tfactory=fractions_by_tfactory,
            output_path=outputs_dir / "rz_injection_fraction.pdf",
        )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--benchmarks-dir", default="benchmarks", type=str)
    parser.add_argument("--outputs-dir", default="outputs", type=str)
    parser.add_argument("--csv-name", default="benchmark_stats.csv", type=str)
    parser.add_argument("--lookup-pkl", default=None)
    parser.add_argument("--factory-distance", default=7, type=int)
    parser.add_argument(
        "--num-factories-sweep",
        default="1-10",
        type=str,
        help="INJEQT sweep values, e.g. '1-10' or '1,2,4,8'. Default: 1-10.",
    )
    parser.add_argument(
        "--parallel-cores",
        default=4,
        type=int,
        help="Number of parallel worker processes for benchmark/config jobs.",
    )
    parser.add_argument(
        "--num-trials",
        default=3,
        type=int,
        help="Number of trials for stochastic configurations.",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Optional seed for reproducible per-run RNG streams.",
    )
    parser.add_argument(
        "--synthesis-epsilon",
        default=None,
        type=float,
        help="Optional synthesis precision (epsilon). If omitted, computed by compute_synthesis_epsilon().",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots from existing CSV data, "
        "skipping benchmark execution/retries.",
    )
    args = parser.parse_args()

    if args.num_trials <= 0:
        raise ValueError("--num-trials must be a positive integer.")
    if args.parallel_cores <= 0:
        raise ValueError("--parallel-cores must be a positive integer.")
    if args.synthesis_epsilon is not None and args.synthesis_epsilon <= 0:
        raise ValueError("--synthesis-epsilon must be positive when provided.")
    num_factories_sweep = parse_num_factories_sweep(args.num_factories_sweep)

    root = Path(__file__).resolve().parent.parent
    benchmarks_dir = (root / args.benchmarks_dir).resolve()
    base_outputs_dir = (root / args.outputs_dir).resolve()
    seed_root = (
        args.seed
        if args.seed is not None
        else int(default_rng().integers(0, 2**63 - 1))
    )
    outputs_dir = base_outputs_dir / str(seed_root)
    csv_path = outputs_dir / args.csv_name

    if not args.plot_only:
        if not benchmarks_dir.exists():
            raise FileNotFoundError(f"Benchmarks directory not found: {benchmarks_dir}")

        if args.lookup_pkl is not None:
            with open(Path(args.lookup_pkl).expanduser().resolve(), "rb") as f:
                lookup = load(f)
        else:
            lookup = load_lookup_table(root)

        existing_rows_by_key = read_existing_rows(csv_path)
        benchmarks = sorted(benchmarks_dir.glob("*.pkl"))
        configs = all_configs(num_factories_sweep)

        jobs: list[RunJob] = []
        for benchmark_path in benchmarks:
            benchmark_name = benchmark_path.name
            circuit = load_benchmark(benchmark_path)
            num_program_bits = infer_num_program_bits(circuit)
            num_noncliffords = count_noncliffords(circuit)
            resolved_synthesis_epsilon = args.synthesis_epsilon
            if resolved_synthesis_epsilon is None:
                resolved_synthesis_epsilon = compute_synthesis_epsilon(
                    GrossCodeErrorModel(),
                    num_noncliffords,
                )
            print(
                f"Queued benchmark: {benchmark_name} ({num_program_bits} program bits)"
            )

            for config in configs:
                trials = args.num_trials if config.stochastic else 1
                pending_trials: list[int] = []
                seed_by_trial: list[int | None] = []
                for trial in range(trials):
                    cache_key = (
                        benchmark_name,
                        config.model,
                        config.rz_factory,
                        config.t_factory_type,
                        config.num_factories,
                        trial,
                        str(resolved_synthesis_epsilon),
                    )
                    cached_row = existing_rows_by_key.get(cache_key)
                    if cached_row is not None:
                        cached_status = (cached_row.get("status") or "").lower()
                        if cached_status == "ok":
                            continue

                    seed_by_trial.append(
                        run_seed(seed_root, "|".join(map(str, cache_key)))
                        if config.stochastic
                        else None
                    )
                    pending_trials.append(trial)

                if len(pending_trials) == 0:
                    continue
                jobs.append(
                    RunJob(
                        benchmark_name=benchmark_name,
                        benchmark_path=str(benchmark_path),
                        num_program_bits=num_program_bits,
                        config=config,
                        trials=tuple(pending_trials),
                        seed_by_trial=tuple(seed_by_trial),
                        factory_distance=args.factory_distance,
                        synthesis_epsilon=resolved_synthesis_epsilon,
                    )
                )

        print(
            f"Executing {len(jobs)} uncached/retry jobs with {args.parallel_cores} workers "
            f"(seed={seed_root}, sweep={num_factories_sweep})"
        )
        new_rows: list[dict[str, Any]] = []
        if len(jobs) > 0:
            if args.parallel_cores == 1 or len(jobs) == 1:
                _init_worker(lookup)
                for job in jobs:
                    new_rows.extend(run_job(job))
            else:
                max_workers = min(args.parallel_cores, len(jobs))
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_init_worker,
                    initargs=(lookup,),
                ) as executor:
                    futures = [executor.submit(run_job, job) for job in jobs]
                    for future in as_completed(futures):
                        new_rows.extend(future.result())

        merged_rows_by_key: dict[
            tuple[str, str, str, str, int, int, str], dict[str, Any]
        ] = {key: dict(value) for key, value in existing_rows_by_key.items()}
        for row in new_rows:
            merged_rows_by_key[row_cache_key(row)] = row

        all_rows = list(merged_rows_by_key.values())
        all_rows.sort(key=row_sort_key)
        write_rows(csv_path, all_rows)
        print(f"Wrote CSV: {csv_path}")

    plot_from_csv(csv_path, outputs_dir)

    print(f"Wrote box plots to: {outputs_dir}")


if __name__ == "__main__":
    main()
