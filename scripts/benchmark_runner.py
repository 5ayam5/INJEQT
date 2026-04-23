from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from csv import DictReader, DictWriter
from dataclasses import dataclass
from hashlib import blake2b
from math import ceil
from pathlib import Path
from pickle import load
from sys import path as sys_path
from typing import Any

from benchmark_config import (
    BASE_COLUMNS,
    FACTORY_TYPES,
    STATS_COLUMNS,
    TDG_FACTORY_TYPES,
    row_t_factory_type,
    rz_factories_for_t_factory_type,
)
from numpy.random import Generator, default_rng

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys_path:
    sys_path.insert(0, str(SRC_DIR))

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
        rz_factories = rz_factories_for_t_factory_type(t)
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
    t_factory_type = row_t_factory_type(row)
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
        canonical["t_factory_type"] = row_t_factory_type(canonical)
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
    t_factory_type = row_t_factory_type(row)
    return (
        str(row["benchmark"]),
        str(row["model"]),
        str(row["rz_factory"]),
        t_factory_type,
        int(row.get("num_factories", 0) or 0),
        int(row["trial"]),
        str(row.get("synthesis_epsilon", "") or ""),
    )


def execute_benchmarks(
    benchmarks_dir: Path,
    csv_path: Path,
    lookup_pkl: str | None,
    factory_distance: int,
    num_factories_sweep: list[int],
    parallel_cores: int,
    num_trials: int,
    seed_root: int,
    synthesis_epsilon: float | None,
) -> None:
    if not benchmarks_dir.exists():
        raise FileNotFoundError(f"Benchmarks directory not found: {benchmarks_dir}")

    if lookup_pkl is not None:
        with open(Path(lookup_pkl).expanduser().resolve(), "rb") as f:
            lookup = load(f)
    else:
        lookup = load_lookup_table(ROOT_DIR)

    existing_rows_by_key = read_existing_rows(csv_path)
    benchmarks = sorted(benchmarks_dir.glob("*.pkl"))
    configs = all_configs(num_factories_sweep)

    jobs: list[RunJob] = []
    for benchmark_path in benchmarks:
        benchmark_name = benchmark_path.name
        circuit = load_benchmark(benchmark_path)
        num_program_bits = infer_num_program_bits(circuit)
        num_noncliffords = count_noncliffords(circuit)
        resolved_synthesis_epsilon = synthesis_epsilon
        if resolved_synthesis_epsilon is None:
            resolved_synthesis_epsilon = compute_synthesis_epsilon(
                GrossCodeErrorModel(),
                num_noncliffords,
            )
        print(f"Queued benchmark: {benchmark_name} ({num_program_bits} program bits)")

        for config in configs:
            trials = num_trials if config.stochastic else 1
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
                    factory_distance=factory_distance,
                    synthesis_epsilon=resolved_synthesis_epsilon,
                )
            )

    print(
        f"Executing {len(jobs)} uncached/retry jobs with {parallel_cores} workers "
        f"(seed={seed_root}, sweep={num_factories_sweep})"
    )

    new_rows: list[dict[str, Any]] = []
    if len(jobs) > 0:
        if parallel_cores == 1 or len(jobs) == 1:
            _init_worker(lookup)
            for job in jobs:
                new_rows.extend(run_job(job))
        else:
            max_workers = min(parallel_cores, len(jobs))
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
