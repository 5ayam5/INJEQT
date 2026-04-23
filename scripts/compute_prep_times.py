from __future__ import annotations

import math
from argparse import ArgumentParser
from statistics import mean
from typing import List

from benchmark_runner import _compute_effective_distance
from ExecutionModels import (
    CultivationFactory,
    STARSurfaceCodeFactory,
)
from numpy.random import default_rng


def _mean_finite(values: List[float]) -> float:
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return math.nan
    return mean(vals)


def main() -> None:
    parser = ArgumentParser(
        description="Compute average prep times for Rz and Cultivation factories"
    )
    parser.add_argument("--physical-error-rate", "-p", default=1e-4, type=float)
    parser.add_argument("--code-distance", "-d", default=7, type=int)
    parser.add_argument(
        "--samples",
        "-n",
        default=100_000,
        type=int,
        help="Number of samples to average for cultivation (t_prep_time is stochastic)",
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="RNG seed for reproducibility"
    )

    args = parser.parse_args()

    phys_rate = float(args.physical_error_rate)
    code_distance = int(args.code_distance)
    samples = int(args.samples)
    seed = int(args.seed)

    # compute effective distances consistent with benchmark_runner usage
    eff_rz_d = _compute_effective_distance(code_distance, "Distillation")
    eff_cult_d = _compute_effective_distance(code_distance, "Cultivation")

    # Use STAR as the RZ factory and sample its prep time (stochastic)
    try:
        star_factory = STARSurfaceCodeFactory(
            d_factory=eff_rz_d,
            physical_qubit_error_rate=phys_rate,
            rng=default_rng(seed + 1),
        )
    except Exception:
        star_times: List[float] = [math.nan]
    else:
        star_times: List[float] = []
        for _ in range(samples):
            try:
                star_times.append(float(star_factory.factory_prep_time))
            except Exception:
                star_times.append(math.nan)

    avg_rz = _mean_finite(star_times)

    # Cultivation: instantiate once and sample its t_prep_time repeatedly
    try:
        cult_factory = CultivationFactory(
            d_factory=eff_cult_d,
            physical_qubit_error_rate=phys_rate,
            rng=default_rng(seed),
        )
    except Exception:
        cult_times = [math.nan]
    else:
        cult_times: List[float] = []
        for _ in range(samples):
            try:
                cult_times.append(float(cult_factory.t_prep_time))
            except Exception:
                cult_times.append(math.nan)

    avg_cult = _mean_finite(cult_times)

    # Print results
    print(
        f"STAR RZ factory average prep time (sampled {samples} times, d_effective={eff_rz_d}): {avg_rz if not math.isnan(avg_rz) else 'N/A'}"
    )
    print()
    print(
        f"Cultivation t_prep_time average (sampled {samples} times, d_effective={eff_cult_d}): {avg_cult if not math.isnan(avg_cult) else 'N/A'}"
    )


if __name__ == "__main__":
    main()
