# INJEQT

Code for **INJEQT: Improved Magic-State Injection Protocol for Fault-Tolerant
Quantum Extractor Architectures**.

If you use this code, please cite:

```bibtex
@misc{sethi2026injeqtimprovedmagicstateinjection,
      title={INJEQT: Improved Magic-State Injection Protocol for Fault-Tolerant Quantum Extractor Architectures},
      author={Sayam Sethi and Sahil Khan and Aditi Awasthi and Abhinav Anand and Jonathan Mark Baker},
      year={2026},
      eprint={2604.25094},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2604.25094},
}
```

Paper: [arXiv:2604.25094](https://arxiv.org/abs/2604.25094)

## Overview

The code models the execution of Pauli-product-rotation circuits on a modular
fault-tolerant architecture built from $[[144,12,12]]$ gross-code "extractor"
modules connected in a line topology, and compares two ways of realising the
non-Clifford rotations:

- **TDG (Tour de Gross, baseline)** — synthesise each $R_z(\theta)$ from a
  sequence of $T$ gates, each supplied by a $T$ factory (distillation or
  cultivation).
- **INJEQT (this work)** — inject $R_z(\theta)$ states directly using
  repeat-until-success angle-doubling, drawing from a bank of `num_factories`
  parallel $R_z$ factories with angle reuse.

For each (benchmark, execution model, factory configuration) the simulator
reports logical error, qubit count, active time, wall-clock time, and space-time
volume; the plotting layer turns those into the figures used in the paper.

## Repository layout

```
src/ExecutionModels.py     factory + execution-model hierarchy (the physics/resource models)
src/experiments.py         circuit evaluation loop and single-run CLI
scripts/benchmark_config.py    shared config: factory types, CSV schema, label helpers
scripts/benchmark_runner.py    parallel sweep driver with CSV-level caching
scripts/benchmark_plotter.py   all figure generation from the results CSV
scripts/run_all_benchmarks.py  top-level entry point (run + plot)
scripts/compute_prep_times.py  utility: average factory prep times / discard rates
benchmarks/                26 compiled benchmark circuits (Git LFS)
imports/magic-state-cultivation  submodule: Gidney et al. cultivation circuits + sinter samplers
outputs/<seed>/            results CSV and figures for a given seed
```

### Models

`src/ExecutionModels.py` defines the factory hierarchy:

| Class                              | Role                                                                                                                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DistillationFactory`              | surface-code $T$ factory; error rate and prep time from lookup tables (d = 7, 9, 11)                                                                                |
| `CultivationFactory`               | magic-state cultivation $T$ factory; stage-1/2 and stage-3 success probabilities measured with `sinter` on circuits from the submodule, then sampled stochastically |
| `LatticeSurgerySurfaceCodeFactory` | $R_z$ factory building the rotation from $T$ states via lattice surgery                                                                                             |
| `TransversalSurfaceCodeFactory`    | $R_z$ factory using transversal gates + correlated decoding                                                                                                         |
| `STARSurfaceCodeFactory`           | STAR-style direct $R_z$ preparation via $[[4,1,1,2]]$ ancillae plus expansion                                                                                       |
| `ColourCodeFactory`                | placeholder; not implemented (rows for it appear as `error` in the CSV)                                                                                             |

and the two execution models: `TDGExecutionModel` (charges
`num_t_injections × (prep + inter-module step)` per rotation) and
`INJEQTExecutionModel` (angle-doubling repeat-until-success against a pool of
factories, tracking per-factory availability times and the angle each factory
currently holds).

## Setup

```bash
git clone --recurse-submodules https://github.com/5ayam5/INJEQT.git
cd INJEQT
git lfs pull            # benchmarks/ and measurement_table.cbor are LFS objects
```

Python 3.12+ is required (the code uses `typing.Self` and `override`). Install
the dependencies of this project plus those of the cultivation submodule:

```bash
pip install numpy matplotlib cbor2 typing_extensions
pip install -r imports/magic-state-cultivation/requirements.txt
```

`stim`, `sinter`, `chromobius`, and `pymatching` (from the submodule
requirements) are only needed for cultivation configurations, which invoke
`sinter collect` to measure cultivation success probabilities. Results are
cached in `sinter_outputs/`, so this cost is paid once per
`(d_colour_code, d_factory, r1, r2, p)` combination.

### Measurement lookup table

In-module Pauli-product measurement costs come from a precomputed table. The
first run builds `meas_lookup_table.pkl` from the LFS-tracked
`measurement_table.cbor` (~750 MB), which takes a while and needs `cbor2`;
subsequent runs load the pickle directly. Pass `--lookup-pkl` to point at an
existing pickle instead.

## Running

Full sweep plus figures:

```bash
python scripts/run_all_benchmarks.py \
    --num-factories-sweep 1-10 \
    --parallel-cores 8 \
    --num-trials 3 \
    --seed 0
```

Results land in `outputs/<seed>/benchmark_stats.csv` alongside the PDFs. The
runner is **incremental**: rows already present with `status=ok` for the same
`(benchmark, model, rz_factory, t_factory_type, num_factories, trial, synthesis_epsilon)`
key are reused, so re-running only executes missing or previously failed
configurations. Seeds are derived deterministically per row (`blake2b` of the
seed root and the cache key), so a given `--seed` reproduces the same numbers.

Useful flags:

| Flag                    | Default                                                      | Meaning                                                                               |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| `--num-factories-sweep` | `1-10`                                                       | INJEQT factory-count sweep; accepts ranges and lists (`1-10`, `1,2,4,8`)              |
| `--factory-distance`    | `7`                                                          | factory code distance (cultivation is bumped to an odd $d \ge 2 d_{\mathrm{colour}}$) |
| `--num-trials`          | `3`                                                          | trials per stochastic configuration                                                   |
| `--seed`                | `0`                                                          | seed root; also names the output directory                                            |
| `--synthesis-epsilon`   | auto                                                         | rotation-synthesis precision $\varepsilon$; overrides `compute_synthesis_epsilon()`   |
| `--parallel-cores`      | `4`                                                          | worker processes                                                                      |
| `--plot-only`           | off                                                          | regenerate figures from the existing CSV                                              |
| `--plot-metrics`        | `total_error,space_time,wall_clock_time,num_physical_qubits` | metrics in the combined figures                                                       |

Regenerate only the plots:

```bash
python scripts/run_all_benchmarks.py --plot-only --seed 0
```

Single configuration, printed to stdout:

```bash
python src/experiments.py \
    benchmarks/official_compiled_circ_RzTranspiled_adder_n118.qasm_after_commute.pkl \
    118 LatticeSurgeryRz --t-factory-type Cultivation --num-factories 4 --seed 0
```

The positional `factory_type` is one of `Distillation`, `Cultivation` (TDG
baselines) or `LatticeSurgeryRz`, `TransversalRz`, `STARRz` (INJEQT).

Average factory prep times and discard rates:

```bash
python scripts/compute_prep_times.py -p 1e-4 -d 7 -n 100000
```

## Outputs

`benchmark_stats.csv` carries one row per run with the configuration columns, a
`status`/`error` pair, and the metrics `#in_modules`, `#inter_modules`,
`in_error`, `inter_error`, `rz_injection_error`, `total_error`, `num_gates`,
`time_in_module`, `time_inter_module`, `time_rz_injection`, `active_time`,
`wall_clock_time`, `num_physical_qubits`, `space_time`. Times are in units of
physical gate steps.

Figures written per seed directory:

- `boxplot_relative_<metric>_vs_tdg_<tfactory>.pdf` — per-benchmark improvement
  over the TDG baseline for INJEQT$^*$ (the best `num_factories` per benchmark,
  chosen by mean improvement).
- `sweep_relative_<metric>_<tfactory>.pdf` — improvement as a function of
  `num_factories`.
- `selected_num_factories_<metric>.pdf` — distribution of the `num_factories`
  INJEQT$^*$ selects.
- `rz_injection_fraction.pdf` — fraction of runtime spent on $R_z$ injection,
  TDG vs INJEQT.
- `boxplot_combined_*.pdf`, `sweep_combined_*.pdf`,
  `selected_num_factories_combined.pdf` — multi-metric versions of the above
  (the figures used in the paper).

`outputs/0/` and `outputs/42/` hold the committed results for the two seeds
reported in the paper.

## Benchmarks

26 circuits from QASMBench, Rz-transpiled and commuted into
Pauli-product-rotation form: `adder`, `dnn`, `ising`, `knn`, `multiplier`,
`qft`, `qugan`, and `wstate` at sizes from 28 to 433 qubits. Each pickle is a
`CompiledCirc` with `uncompiled_operations` and `compiled_operations`, where an
operation is `(pauli_strings, qubit_indices, gate_type[, angle])` and
`gate_type` is `"clifford"` or `"nonclifford"`. Program qubits are mapped to
modules 11 at a time (`NUM_LOGICAL_QUBITS_PER_MODULE = 11`).

## Acknowledgements

Cultivation circuits and `sinter` decoders come from
[Strilanc/magic-state-cultivation](https://github.com/Strilanc/magic-state-cultivation),
vendored as a submodule under `imports/`.

## A note on the committed outputs

The CSVs and figures under `outputs/` were generated incrementally over time,
and the runner reuses any row already marked `status=ok` (its cache key covers
the configuration but not the code version). Some rows therefore predate later
refinements to the timing and error models, so individual numbers may differ
slightly from what the current code produces. The trends and relative
comparisons the figures are used for are unaffected.

To regenerate from scratch, delete the target
`outputs/<seed>/benchmark_stats.csv` first — otherwise the cached rows are
carried over.
