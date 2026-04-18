from __future__ import annotations

import argparse
import math
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from numpy.random import Generator, default_rng

try:
    import cbor2
except ImportError:  # optional until lookup table build is needed
    cbor2 = None

from ExecutionModels import (
    CultivationFactory,
    DistillationFactory,
    ExecutionModel,
    INJEQTExecutionModel,
    NeutralAtomSurfaceCodeFactory,
    STARSurfaceCodeFactory,
    SuperconductingSurfaceCodeFactory,
    TDGExecutionModel,
)

RotationsType = list[
    tuple[list[str], list[str], str] | tuple[list[str], list[str], str, str]
]


class CompiledCirc:
    uncompiled_operations: RotationsType
    compiled_operations: RotationsType


NUM_LOGICAL_QUBITS_PER_MODULE = 11


@dataclass(frozen=True)
class GrossCodeErrorModel:
    in_cost: float = 10**-9
    inter_cost: float = 10**-7.3
    t_injection_cost: float = 10**-7.4


def build_lookup(measurements: list[dict]) -> dict[int, int]:
    d: dict[int, int] = {}
    number = 0
    for entry in measurements:
        c = entry["cost"]
        d[number] = c
        number += 1
    return d


def load_lookup_table(root: Path) -> dict[int, int]:
    pkl_path = root / "meas_lookup_table.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pkl.load(f)

    cbor_path = root / "measurement_table.cbor"
    if not cbor_path.exists():
        raise FileNotFoundError(
            "Need meas_lookup_table.pkl or measurement_table.cbor in project root."
        )
    if cbor2 is None:
        raise ImportError(
            "cbor2 is required to build meas_lookup_table.pkl from measurement_table.cbor"
        )

    with open(cbor_path, "rb") as f:
        data = cbor2.load(f)
    measurements = data["measurements"]
    lookup = build_lookup(measurements)
    with open(pkl_path, "wb") as f:
        pkl.dump(lookup, f)
    return lookup


def pauli_to_mask(paulis, indices) -> int:
    assert len(paulis) == len(indices), "paulis and indices must match in length"

    x_mask = 0
    z_mask = 0
    indices = [int(x) for x in indices]
    for i in range(len(paulis)):
        p = paulis[i]
        index = indices[i]
        assert index <= NUM_LOGICAL_QUBITS_PER_MODULE
        if p == "I":
            continue
        elif p == "X":
            x_mask |= 1 << index
        elif p == "Z":
            z_mask |= 1 << index
        elif p == "Y":
            x_mask |= 1 << index
            z_mask |= 1 << index
        else:
            raise ValueError(f"Invalid Pauli: {p}")

    mask = (z_mask << 12) | x_mask
    assert 0 <= mask <= (1 << (2 * NUM_LOGICAL_QUBITS_PER_MODULE + 2)) - 1, (
        "Mask out of bounds"
    )
    return mask


def find_fixed_partition_original(
    lookup: dict[int, int],
    orig_measurement: int,
) -> int:
    pivots = ["X", "Y", "Z"]
    assert NUM_LOGICAL_QUBITS_PER_MODULE == 11, (
        "This method currently assumes 11 logical qubits per module."
    )

    def cost_computation(num_gates: int) -> int:
        if num_gates == 0:
            return 0
        return 3 * num_gates - 2

    orig_meas_map: dict[int, str] = {}
    i = 0
    num = orig_measurement
    while i < 12:
        bit_x = (num >> i) & 1
        bit_z = (num >> (i + 12)) & 1
        if bit_x == 1 and bit_z == 1:
            orig_meas_map[i] = "Y"
        elif bit_x == 1 and bit_z == 0:
            orig_meas_map[i] = "X"
        elif bit_x == 0 and bit_z == 1:
            orig_meas_map[i] = "Z"
        else:
            orig_meas_map[i] = "I"
        i += 1

    for i in orig_meas_map.keys():
        if i >= NUM_LOGICAL_QUBITS_PER_MODULE:
            assert orig_meas_map[i] == "I"

    cost = math.inf
    for pivot_choice in pivots:
        pauli_list_1 = []
        whole_set = list(set(range(NUM_LOGICAL_QUBITS_PER_MODULE)))
        for x in whole_set:
            pauli_list_1.append(orig_meas_map[x])
        pauli_list_1.append(pivot_choice)
        whole_set.append(NUM_LOGICAL_QUBITS_PER_MODULE)
        candidate_mask = pauli_to_mask(pauli_list_1, whole_set)
        cost1 = cost_computation(lookup[candidate_mask])
        if cost1 < cost:
            cost = cost1

    assert cost != math.inf
    return int(cost)


def default_input_mapping(num_program_bits: int) -> dict[int, int]:
    # program qubit i -> module floor(i / NUM_LOGICAL_QUBITS_PER_MODULE)
    return {i: i // NUM_LOGICAL_QUBITS_PER_MODULE for i in range(num_program_bits)}


def find_chunking_from_map(
    pauli_strings: list[str], bits: list[str], num_modules: int, mapping: dict[int, int]
) -> tuple[list[list[str]], list[list[int]]]:
    p_chunks = [[] for _ in range(num_modules)]
    b_chunks = [[] for _ in range(num_modules)]
    for i in range(len(pauli_strings)):
        p = pauli_strings[i]
        b = int(bits[i])
        module_num = mapping[b]
        p_chunks[module_num].append(p)
        b_chunks[module_num].append(b)
    return p_chunks, b_chunks


def create_and_map_rotations(
    rotations: RotationsType, num_program_bits: int
) -> tuple[list[tuple[list[int], str] | tuple[list[int], str, str]], list[list[int]]]:
    all_rotations_binaries = []
    num_modules = math.ceil(num_program_bits / NUM_LOGICAL_QUBITS_PER_MODULE)
    mapping = default_input_mapping(num_program_bits)

    associated_modules = []
    for rotation in rotations:
        unparsed_pauli_strings = rotation[0]
        pauli_strings = [p.strip("-") for p in unparsed_pauli_strings]
        bits = rotation[1]

        chunks_pauli_strings, chunks_bits_affected = find_chunking_from_map(
            pauli_strings,
            bits,
            num_modules=num_modules,
            mapping=mapping,
        )

        assert len(chunks_pauli_strings) == len(chunks_bits_affected)
        this_rotations_binaries = []
        this_rotations_assoc_modules = []
        for c in range(len(chunks_pauli_strings)):
            chunk_string = chunks_pauli_strings[c]
            chunk_bits = chunks_bits_affected[c]
            real_bits = []
            for x in chunk_bits:
                x = int(x)
                assert 0 <= x < num_program_bits
                real_x = x % NUM_LOGICAL_QUBITS_PER_MODULE
                real_bits.append(real_x)
            assert len(chunk_bits) == len(real_bits)
            binary = pauli_to_mask(
                paulis=chunk_string,
                indices=real_bits,
            )
            if binary != 0:
                this_rotations_binaries.append(binary)
                this_rotations_assoc_modules.append(c)

        if len(this_rotations_binaries) == 0:
            assert len(this_rotations_assoc_modules) == 0
            continue

        if rotation[2] == "nonclifford":
            assert len(rotation) == 4, "Nonclifford rotations must include angle"
            all_rotations_binaries.append(
                (this_rotations_binaries, rotation[2], rotation[3])
            )
        else:
            all_rotations_binaries.append((this_rotations_binaries, rotation[2]))
        associated_modules.append(this_rotations_assoc_modules)

    assert len(all_rotations_binaries) == len(associated_modules)
    return all_rotations_binaries, associated_modules


def line_topology_inter_cost(
    modules: Sequence[int], source_modules: Sequence[int]
) -> tuple[int, list[int]]:
    if len(modules) == 0:
        return 0, []
    best, low, high = math.inf, max(modules), min(modules)
    for s in source_modules:
        lo = min([s, *modules])
        hi = max([s, *modules])
        if hi - lo < best:
            best = hi - lo
            low = lo
            high = hi
    return int(best if best != math.inf else 0), list(range(low, high + 1))


def spawn_child_rng(rng: Generator) -> Generator:
    return default_rng(int(rng.integers(0, 2**63 - 1)))


def evaluate_circuit(
    circuit: CompiledCirc,
    lookup: dict[int, int],
    num_program_bits: int,
    error_model: GrossCodeErrorModel,
    execution_model: ExecutionModel,
):
    mapped_rotations, associated_modules = create_and_map_rotations(
        circuit.compiled_operations, num_program_bits
    )

    in_modules = 0
    inter_modules = 0
    rz_injection_error = 0

    time_in_module = 0.0
    time_inter_module = 0.0
    time_rz_injection = 0.0

    module_ready: dict[int, float] = {}
    wall_clock = 0.0

    for count in range(len(mapped_rotations)):
        mapped_rotation = mapped_rotations[count]
        if len(mapped_rotation) == 3:
            pauli_product_rotation, gate_type, angle_str = mapped_rotation
            angle = float(eval(angle_str, {"__builtins__": {}}, {"pi": math.pi}))
        else:
            pauli_product_rotation, gate_type = mapped_rotation
            angle = None
        assoc_inter_modules_list = associated_modules[count]

        if gate_type == "nonclifford":
            inter_modules_count, touched_modules = line_topology_inter_cost(
                assoc_inter_modules_list,
                [-1],
            )
        else:
            inter_modules_count, touched_modules = line_topology_inter_cost(
                assoc_inter_modules_list[1:],
                [assoc_inter_modules_list[0]],
            )

        # Phase 1: in-module operations
        in_modules_count_this_step = 0
        for m in pauli_product_rotation:
            res = find_fixed_partition_original(lookup, m)
            in_modules += res
            in_modules_count_this_step = max(in_modules_count_this_step, res)

        # Phase 2: inter-module operations with logarithmic depth
        inter_modules += inter_modules_count
        inter_depth = math.ceil(math.log2(inter_modules_count + 1))

        in_start_t = 0.0
        for m in assoc_inter_modules_list:
            in_start_t = max(in_start_t, module_ready.get(m, 0.0))
        in_time = in_modules_count_this_step * execution_model.in_module_step
        in_end_t = in_start_t + in_time

        inter_start_t = in_end_t
        for m in touched_modules:
            inter_start_t = max(inter_start_t, module_ready.get(m, 0.0))

        inter_time = inter_depth * execution_model.inter_module_layer_step
        inter_end_t = inter_start_t + inter_time
        end_t = inter_end_t

        # Phase 3: Rz synthesis via T injections
        if gate_type == "nonclifford":
            assert angle is not None, "Nonclifford gates must have an angle specified"
            rz_finish_time, num_injections = execution_model.rz_injection_time(
                in_start_t, inter_end_t, angle
            )
            rz_injection_error += (
                num_injections * error_model.inter_cost
                + execution_model.factory.synthesis_logical_error_rate
            )
            time_rz_injection += rz_finish_time - inter_end_t
            end_t = rz_finish_time

        for m in touched_modules:
            if m == -1:
                continue
            module_ready[m] = end_t

        wall_clock = max(wall_clock, end_t)
        time_in_module += in_time
        time_inter_module += inter_time

    in_error = in_modules * error_model.in_cost
    inter_error = inter_modules * error_model.inter_cost
    total_error = in_error + inter_error + rz_injection_error

    num_physical_qubits = execution_model.num_physical_qubits
    space_time = num_physical_qubits * wall_clock

    return {
        "#in_modules": in_modules,
        "#inter_modules": inter_modules,
        "in_error": in_error,
        "inter_error": inter_error,
        "rz_injection_error": rz_injection_error,
        "total_error": total_error,
        "num_gates": len(mapped_rotations),
        "time_in_module": time_in_module,
        "time_inter_module": time_inter_module,
        "time_rz_injection": time_rz_injection,
        "active_time": time_in_module + time_inter_module + time_rz_injection,
        "wall_clock_time": wall_clock,
        "num_physical_qubits": num_physical_qubits,
        "space_time": space_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "compiled_circuit",
        help="Path to compiled circuit pickle (CompiledCirc)",
    )
    parser.add_argument(
        "num_program_bits",
        type=int,
        help="Total number of program qubits in the benchmark",
    )
    parser.add_argument(
        "factory_type",
        choices=[
            "Distillation",
            "Cultivation",
            "SuperconductingRz",
            "NeutralAtomRz",
            "STARRz",
        ],
        type=str,
        help="Type of factory to use for inter-module operations",
    )
    parser.add_argument(
        "--t-factory-type",
        default="Distillation",
        choices=["Distillation", "Cultivation", "ColourCode"],
        type=str,
        help="Type of factory to use for T state production (default: Distillation)",
    )
    parser.add_argument(
        "--num-factories",
        default=1,
        type=int,
        help="Number of factories to use for inter-module operations (if applicable; default: 1)",
    )
    parser.add_argument(
        "--factory-distance",
        default=7,
        type=int,
        help="Code distance to use for factories (if applicable; default: 7)",
    )
    parser.add_argument(
        "--lookup-pkl",
        default=None,
        help="Optional explicit path to meas_lookup_table.pkl",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Optional RNG seed for stochastic factory/timing behavior.",
    )
    args = parser.parse_args()
    root_rng = default_rng(args.seed)

    root = Path(__file__).resolve().parent

    if args.lookup_pkl is not None:
        with open(Path(args.lookup_pkl).expanduser().resolve(), "rb") as f:
            lookup = pkl.load(f)
    else:
        lookup = load_lookup_table(root)

    compiled_path = Path(args.compiled_circuit).expanduser().resolve()
    with open(compiled_path, "rb") as f:
        circuit: CompiledCirc = pkl.load(f)

    num_modules = math.ceil(args.num_program_bits / NUM_LOGICAL_QUBITS_PER_MODULE)

    factory_type = args.factory_type
    if factory_type == "Distillation":
        factory_model = DistillationFactory(args.factory_distance)
        execution_model = TDGExecutionModel(factory_model, num_modules)

    elif factory_type == "Cultivation":
        factory_model = CultivationFactory(
            args.factory_distance,
            rng=spawn_child_rng(root_rng),
        )
        execution_model = TDGExecutionModel(factory_model, num_modules)

    elif factory_type == "SuperconductingRz":
        factory_model = SuperconductingSurfaceCodeFactory(
            args.factory_distance,
            t_factory_type=args.t_factory_type,
            rng=spawn_child_rng(root_rng),
        )
        execution_model = INJEQTExecutionModel(
            factory_model,
            num_modules,
            num_factories=args.num_factories,
            rng=spawn_child_rng(root_rng),
        )

    elif factory_type == "NeutralAtomRz":
        factory_model = NeutralAtomSurfaceCodeFactory(
            args.factory_distance,
            t_factory_type=args.t_factory_type,
            rng=spawn_child_rng(root_rng),
        )
        execution_model = INJEQTExecutionModel(
            factory_model,
            num_modules,
            num_factories=args.num_factories,
            rng=spawn_child_rng(root_rng),
        )

    elif factory_type == "STARRz":
        factory_model = STARSurfaceCodeFactory(
            args.factory_distance,
            rng=spawn_child_rng(root_rng),
        )
        execution_model = INJEQTExecutionModel(
            factory_model,
            num_modules,
            num_factories=args.num_factories,
            rng=spawn_child_rng(root_rng),
        )

    else:
        raise ValueError(f"Invalid factory type: {factory_type}")

    stats = evaluate_circuit(
        circuit=circuit,
        lookup=lookup,
        num_program_bits=args.num_program_bits,
        error_model=GrossCodeErrorModel(),
        execution_model=execution_model,
    )

    print(f"circuit: {compiled_path.name}")
    print(
        "mapping: default_input_mapping (module = qubit // NUM_LOGICAL_QUBITS_PER_MODULE)"
    )
    for k in stats.keys():
        if isinstance(stats[k], int):
            print(f"{k}: {stats[k]}")
        else:
            print(f"{k}: {stats[k]:.3e}")


if __name__ == "__main__":
    main()
