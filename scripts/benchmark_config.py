from __future__ import annotations

from re import match, search
from typing import Any

FACTORY_TYPES = (
    "Distillation",
    "Cultivation",
    "STAR",
)
TDG_FACTORY_TYPES = (
    "Distillation",
    "Cultivation",
)
FACTORY_TYPE_TO_RZ_FACTORIES = {
    "Distillation": ("LatticeSurgery", "Transversal"),
    "Cultivation": ("LatticeSurgery", "Transversal"),
    "STAR": ("STAR",),
}

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
METRIC_LABELS = {
    "wall_clock_time": "Wall-Clock Time",
    "active_time": "Active Time",
    "total_error": "Total Error",
    "num_physical_qubits": "#Physical Qubits",
    "space_time": "Space-Time",
}
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


def normalized_t_factory_type(t_factory_type: str, rz_factory: str) -> str:
    if t_factory_type:
        return t_factory_type
    if rz_factory == "STAR":
        return "STAR"
    return ""


def row_t_factory_type(row: dict[str, str] | dict[str, Any]) -> str:
    return normalized_t_factory_type(
        str(row.get("t_factory_type", "") or ""),
        str(row.get("rz_factory", "") or ""),
    )


def rz_factories_for_t_factory_type(t_factory_type: str) -> tuple[str, ...]:
    return FACTORY_TYPE_TO_RZ_FACTORIES.get(t_factory_type, ())


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
