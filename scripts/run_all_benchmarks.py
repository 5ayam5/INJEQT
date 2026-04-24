from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from sys import path as sys_path

import __main__

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys_path:
    sys_path.insert(0, str(SRC_DIR))
from experiments import (
    CompiledCirc,
)

__main__.CompiledCirc = CompiledCirc  # type: ignore

from benchmark_config import parse_num_factories_sweep
from benchmark_plotter import plot_from_csv
from benchmark_runner import execute_benchmarks
from numpy.random import default_rng


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
    parser.add_argument(
        "--plot-metrics",
        default="total_error,space_time,wall_clock_time,num_physical_qubits",
        type=str,
        help="Comma-separated metrics to plot in combined figure. "
        "Default: total_error,space_time,wall_clock_time,num_physical_qubits",
    )
    args = parser.parse_args()

    if args.num_trials <= 0:
        raise ValueError("--num-trials must be a positive integer.")
    if args.parallel_cores <= 0:
        raise ValueError("--parallel-cores must be a positive integer.")
    if args.synthesis_epsilon is not None and args.synthesis_epsilon <= 0:
        raise ValueError("--synthesis-epsilon must be positive when provided.")
    num_factories_sweep = parse_num_factories_sweep(args.num_factories_sweep)

    plot_metrics = [m.strip() for m in args.plot_metrics.split(",") if m.strip()]
    if not plot_metrics:
        raise ValueError("--plot-metrics must contain at least one metric.")

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
        execute_benchmarks(
            benchmarks_dir=benchmarks_dir,
            csv_path=csv_path,
            lookup_pkl=args.lookup_pkl,
            factory_distance=args.factory_distance,
            num_factories_sweep=num_factories_sweep,
            parallel_cores=args.parallel_cores,
            num_trials=args.num_trials,
            seed_root=seed_root,
            synthesis_epsilon=args.synthesis_epsilon,
        )

    plot_from_csv(csv_path, outputs_dir, plot_metrics)
    print(f"Wrote plots to: {outputs_dir}")


if __name__ == "__main__":
    main()
