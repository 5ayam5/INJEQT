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
from numpy.random import Generator, default_rng

from experiments import (
    NUM_LOGICAL_QUBITS_PER_MODULE,
    CompiledCirc,
    GrossCodeErrorModel,
    evaluate_circuit,
    load_lookup_table,
)
from TimingModels import (
    ColourCodeFactory,
    CultivationFactory,
    DistillationFactory,
    INJEQTExecutionModel,
    NeutralAtomSurfaceCodeFactory,
    STARSurfaceCodeFactory,
    SuperconductingSurfaceCodeFactory,
    TDGExecutionModel,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

T_FACTORY_TYPES = ("Distillation", "Cultivation", "ColourCode")
INJEQT_SWEEP_FACTORIES = ("Superconducting", "NeutralAtom", "STAR")
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
    "status",
    "error",
]


@dataclass(frozen=True)
class RunConfig:
    model: str
    policy: str
    rz_factory: str
    t_factory_type: str | None
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


_LOOKUP_TABLE: dict[int, int] | None = None


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

    deduped: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


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


def build_execution_model(
    config: RunConfig,
    factory_distance: int,
    run_rng: Generator,
    num_modules: int,
):
    if config.model == "TDG":
        if config.t_factory_type == "Distillation":
            factory_model = DistillationFactory(factory_distance)
        elif config.t_factory_type == "Cultivation":
            factory_model = CultivationFactory(
                factory_distance,
                rng=spawn_child_rng(run_rng),
            )
        elif config.t_factory_type == "ColourCode":
            factory_model = ColourCodeFactory(factory_distance)
        else:
            raise ValueError(f"Unknown T factory type: {config.t_factory_type}")
        return TDGExecutionModel(factory_model, num_modules)

    if config.rz_factory == "Superconducting":
        factory_model = SuperconductingSurfaceCodeFactory(
            factory_distance,
            t_factory_type=config.t_factory_type or "Distillation",
            rng=spawn_child_rng(run_rng),
        )
    elif config.rz_factory == "NeutralAtom":
        factory_model = NeutralAtomSurfaceCodeFactory(
            factory_distance,
            t_factory_type=config.t_factory_type or "Distillation",
            rng=spawn_child_rng(run_rng),
        )
    elif config.rz_factory == "STAR":
        factory_model = STARSurfaceCodeFactory(
            factory_distance,
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
    first_trial_failed = False
    first_trial_error = ""

    for index, trial in enumerate(job.trials):
        seed = job.seed_by_trial[index]
        row: dict[str, Any] = {
            "benchmark": job.benchmark_name,
            "num_program_bits": job.num_program_bits,
            "model": job.config.model,
            "policy": job.config.policy,
            "rz_factory": job.config.rz_factory,
            "t_factory_type": job.config.t_factory_type or "",
            "num_factories": job.config.num_factories,
            "trial": trial,
            "seed": seed if seed is not None else "",
            "status": "ok",
            "error": "",
        }

        if first_trial_failed:
            row["status"] = "skipped"
            row["error"] = (
                f"Skipped because first trial in this job failed: {first_trial_error}"
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
            )
            stats = evaluate_circuit(
                circuit=circuit,
                lookup=_LOOKUP_TABLE,
                num_program_bits=job.num_program_bits,
                error_model=GrossCodeErrorModel(),
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
            if index == 0:
                first_trial_failed = True
                first_trial_error = row["error"]

        rows.append(row)

    job_label = (
        job.benchmark_name,
        job.config.policy,
        job.config.rz_factory,
        job.config.t_factory_type or "",
        job.config.num_factories,
    )
    print(f"Completed job {job_label} for trials {job.trials}")
    return rows


def all_configs(num_factories_sweep: list[int]) -> list[RunConfig]:
    configs: list[RunConfig] = []
    for t in T_FACTORY_TYPES:
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

    for t in T_FACTORY_TYPES:
        for num_factories in num_factories_sweep:
            for rz_factory in INJEQT_SWEEP_FACTORIES:
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


def row_cache_key(row: dict[str, Any]) -> tuple[str, str, str, str, int, int]:
    num_factories = int(row.get("num_factories", 0) or 0)
    trial = int(row["trial"])
    return (
        str(row["benchmark"]),
        str(row["model"]),
        str(row["rz_factory"]),
        str(row.get("t_factory_type", "") or ""),
        num_factories,
        trial,
    )


def read_existing_rows(
    csv_path: Path,
) -> dict[tuple[str, str, str, str, int, int], dict[str, str]]:
    if not csv_path.exists():
        return {}

    with open(csv_path, newline="") as f:
        rows = list(DictReader(f))

    latest: dict[tuple[str, str, str, str, int, int], dict[str, str]] = {}
    for row in rows:
        try:
            latest[row_cache_key(row)] = row
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


def row_sort_key(row: dict[str, Any]) -> tuple[str, str, str, str, int, int]:
    return (
        str(row["benchmark"]),
        str(row["model"]),
        str(row["rz_factory"]),
        str(row.get("t_factory_type", "") or ""),
        int(row.get("num_factories", 0) or 0),
        int(row["trial"]),
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
    if baseline_value <= 0:
        return None
    return (baseline_value - candidate_value) / baseline_value * 100.0


def _build_tdg_baseline_map(
    rows: list[dict[str, str]],
    metric: str,
    t_factory_type: str,
) -> dict[tuple[str, int], float]:
    baseline: dict[tuple[str, int], float] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("model") != "TDG":
            continue
        if row.get("t_factory_type") != t_factory_type:
            continue
        raw_value = row.get(metric, "")
        if raw_value == "":
            continue
        baseline[(row["benchmark"], int(row["trial"]))] = float(raw_value)
    return baseline


def _collect_relative_series(
    rows: list[dict[str, str]],
    metric: str,
    tdg_t_factory_type: str,
    candidate_rz_factory: str,
    candidate_t_factory_type: str | None,
) -> dict[str, list[float]]:
    baseline = _build_tdg_baseline_map(rows, metric, tdg_t_factory_type)
    series: dict[str, list[float]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("model") != "INJEQT":
            continue
        if row.get("rz_factory") != candidate_rz_factory:
            continue
        if candidate_t_factory_type is None:
            if row.get("t_factory_type", "") != "":
                continue
        elif row.get("t_factory_type") != candidate_t_factory_type:
            continue

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
    tdg_t_factory_type: str,
    candidate_rz_factory: str,
    candidate_t_factory_type: str,
) -> dict[int, dict[str, list[float]]]:
    baseline = _build_tdg_baseline_map(rows, metric, tdg_t_factory_type)
    series: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("model") != "INJEQT":
            continue
        if row.get("rz_factory") != candidate_rz_factory:
            continue
        if row.get("t_factory_type") != candidate_t_factory_type:
            continue
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


def _plot_grouped_boxplot(
    benchmarks: list[str],
    benchmark_labels: list[str],
    series_data: dict[str, dict[str, list[float]]],
    title: str,
    output_path: Path,
) -> None:
    if len(benchmarks) == 0 or len(series_data) == 0:
        plt.figure(figsize=(10, 4))
        plt.title(title)
        plt.text(0.5, 0.5, "No valid data", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

    labels = list(series_data.keys())
    colors = [
        "#4472C4",
        "#ED7D31",
        "#70AD47",
        "#A5A5A5",
        "#5B9BD5",
        "#FFC000",
        "#264478",
        "#9E480E",
    ]
    x_positions = list(range(len(benchmarks)))
    avg_x_position = len(benchmarks)
    width = 0.75 / max(1, len(labels))
    legend_handles: list[Patch] = []

    plt.figure(figsize=(max(12, len(benchmarks) * 0.6), 6))
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

        color = colors[idx % len(colors)]
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
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"color": color},
            capprops={"color": color},
            boxprops={"edgecolor": "black"},
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
            edgecolor="black",
            alpha=0.85,
        )
        legend_handles.append(Patch(facecolor=color, edgecolor="black", label=label))

    if len(legend_handles) == 0:
        plt.text(0.5, 0.5, "No valid data", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.xticks(
        [*x_positions, avg_x_position],
        [*benchmark_labels, "Average"],
        rotation=35,
        ha="right",
    )
    plt.xlabel("Benchmark")
    plt.ylabel("Relative improvement over TDG (%)")
    plt.title(title)
    plt.legend(handles=legend_handles)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_sweep_summary(
    sweep_series_by_factory: dict[str, dict[int, dict[str, list[float]]]],
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 5))
    has_data = False
    colors = {
        "Superconducting": "#4472C4",
        "NeutralAtom": "#ED7D31",
        "STAR": "#70AD47",
    }

    for rz_factory, series in sweep_series_by_factory.items():
        x_values: list[int] = []
        y_values: list[float] = []
        for num_factories in sorted(series.keys()):
            benchmark_map = series[num_factories]
            values = [
                v for per_benchmark in benchmark_map.values() for v in per_benchmark
            ]
            if len(values) == 0:
                continue
            x_values.append(num_factories)
            y_values.append(sum(values) / len(values))
        if len(x_values) == 0:
            continue
        has_data = True
        plt.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.0,
            color=colors.get(rz_factory, "#70AD47"),
            label=rz_factory,
        )

    if not has_data:
        plt.title(title)
        plt.text(0.5, 0.5, "No valid data", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Number of INJEQT factories")
    plt.ylabel("Global mean relative improvement over TDG (%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_from_csv(csv_path: Path, outputs_dir: Path) -> None:
    rows = load_rows(csv_path)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    for metric in PLOT_COLUMNS:
        for t_factory_type in T_FACTORY_TYPES:
            sweep_series_by_factory: dict[str, dict[int, dict[str, list[float]]]] = {}
            best_series_by_factory: dict[str, dict[str, list[float]]] = {}
            for rz_factory in INJEQT_SWEEP_FACTORIES:
                sweep = _collect_sweep_series(
                    rows=rows,
                    metric=metric,
                    tdg_t_factory_type=t_factory_type,
                    candidate_rz_factory=rz_factory,
                    candidate_t_factory_type=t_factory_type,
                )
                sweep_series_by_factory[rz_factory] = sweep
                best, _ = _compute_injeqt_star_series(sweep)
                best_series_by_factory[rz_factory] = best

            # union of benchmarks across factories
            benchmark_order = sorted(
                set().union(*(bs.keys() for bs in best_series_by_factory.values()))
            )
            benchmark_labels = [
                benchmark_display_label(benchmark) for benchmark in benchmark_order
            ]
            series_data = {
                f"INJEQT$^*$ {rz}": best_series_by_factory.get(rz, {})
                for rz in INJEQT_SWEEP_FACTORIES
            }

            _plot_grouped_boxplot(
                benchmarks=benchmark_order,
                benchmark_labels=benchmark_labels,
                series_data=series_data,
                title=(f"{metric}: Relative improvement over TDG ({t_factory_type})"),
                output_path=(
                    outputs_dir
                    / f"boxplot_relative_{metric}_vs_tdg_{t_factory_type.lower()}.png"
                ),
            )

            _plot_sweep_summary(
                sweep_series_by_factory={
                    rz: sweep_series_by_factory.get(rz, {})
                    for rz in INJEQT_SWEEP_FACTORIES
                },
                title=f"{metric}: INJEQT sweep over num_factories ({t_factory_type})",
                output_path=outputs_dir
                / f"sweep_relative_{metric}_{t_factory_type.lower()}.png",
            )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--benchmarks-dir", default="benchmarks", type=str)
    parser.add_argument("--outputs-dir", default="outputs", type=str)
    parser.add_argument("--csv-name", default="benchmark_stats.csv", type=str)
    parser.add_argument("--lookup-pkl", default=None)
    parser.add_argument("--factory-distance", default=7, type=int)
    parser.add_argument(
        "--num-factories",
        default=1,
        type=int,
        help="Factory count for STAR policy (default: 1).",
    )
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
    args = parser.parse_args()

    if args.num_trials <= 0:
        raise ValueError("--num-trials must be a positive integer.")
    if args.num_factories <= 0:
        raise ValueError("--num-factories must be a positive integer.")
    if args.parallel_cores <= 0:
        raise ValueError("--parallel-cores must be a positive integer.")
    num_factories_sweep = parse_num_factories_sweep(args.num_factories_sweep)

    root = Path(__file__).resolve().parent
    benchmarks_dir = (root / args.benchmarks_dir).resolve()
    base_outputs_dir = (root / args.outputs_dir).resolve()
    seed_root = (
        args.seed
        if args.seed is not None
        else int(default_rng().integers(0, 2**63 - 1))
    )
    outputs_dir = base_outputs_dir / str(seed_root)
    csv_path = outputs_dir / args.csv_name

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
        print(f"Queued benchmark: {benchmark_name} ({num_program_bits} program bits)")

        for config in configs:
            trials = args.num_trials if config.stochastic else 1
            pending_trials: list[int] = []
            seed_by_trial: list[int | None] = []
            for trial in range(trials):
                cache_key = (
                    benchmark_name,
                    config.model,
                    config.rz_factory,
                    config.t_factory_type or "",
                    config.num_factories,
                    trial,
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

    merged_rows_by_key: dict[tuple[str, str, str, str, int, int], dict[str, Any]] = {
        key: dict(value) for key, value in existing_rows_by_key.items()
    }
    for row in new_rows:
        merged_rows_by_key[row_cache_key(row)] = row

    all_rows = list(merged_rows_by_key.values())
    all_rows.sort(key=row_sort_key)
    write_rows(csv_path, all_rows)
    plot_from_csv(csv_path, outputs_dir)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote box plots to: {outputs_dir}")


if __name__ == "__main__":
    main()
