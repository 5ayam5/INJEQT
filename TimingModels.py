from abc import abstractmethod
from csv import DictReader
from dataclasses import dataclass, field
from math import ceil, fmod, log, log10, pi
from os import environ
from pathlib import Path
from subprocess import run
from sys import path
from typing import Self

from numpy.random import Generator, default_rng
from typing_extensions import override

__ROOT_DIR__ = Path(__file__).parent
CULTIVATION_SRC = __ROOT_DIR__ / "imports" / "magic-state-cultivation"
SINTER_SRC = CULTIVATION_SRC / "src"
if str(SINTER_SRC.resolve()) not in path:
    path.insert(0, str(SINTER_SRC.resolve()))

import cultiv
import gen

MAKE_CIRCUITS = CULTIVATION_SRC / "tools" / "make_circuits"
SINTER_OUTPUTS_DIR = __ROOT_DIR__ / "sinter_outputs"

"""
Different kinds of factories
1. T Factory
    a. Distillation
    b. Cultivation
2. Rz Factory
    a. Superconducting qubits (operates via lattice surgery)
    b. Neutral atoms (operates via transversal gates and correlated decoding)
    a. STAR/transversal synthesis of Rz states
    c. Color code factory (can prepare T and Rz states directly)
"""


@dataclass(frozen=True)
class Factory:
    d_factory: int
    physical_qubit_error_rate: float = 1e-4

    @property
    @abstractmethod
    def synthesis_logical_error_rate(self: Self) -> float:
        pass

    @property
    @abstractmethod
    def factory_syndrome_extraction_cycles(self: Self) -> int:
        pass

    @property
    @abstractmethod
    def factory_prep_time(self: Self) -> float:
        pass


@dataclass(frozen=True)
class TFactory(Factory):
    synthesis_epsilon: float = 1e-10
    num_t_injections: int = ceil(-10 * log10(synthesis_epsilon))

    @property
    @abstractmethod
    def t_prep_time(self: Self) -> float:
        pass

    @property
    @override
    def factory_prep_time(self: Self) -> float:
        return self.t_prep_time


@dataclass(frozen=True)
class SurfaceCodeFactory(Factory):
    @property
    @override
    def factory_syndrome_extraction_cycles(self: Self) -> int:
        return 6


@dataclass(frozen=True)
class DistillationFactory(TFactory, SurfaceCodeFactory):
    @property
    @override
    def synthesis_logical_error_rate(self: Self) -> float:
        if self.physical_qubit_error_rate != 1e-4:
            raise NotImplementedError(
                "Only physical qubit error rate of 1e-4 is supported for now."
            )

        return {
            7: 4.4e-8,
            9: 9.3e-10,
            11: 1.9e-11,
        }[self.d_factory]

    @property
    @override
    def t_prep_time(self: Self) -> float:
        def distillation_time_lookup() -> float:
            if self.physical_qubit_error_rate != 1e-4:
                raise NotImplementedError(
                    "Only physical qubit error rate of 1e-4 is supported for now."
                )

            return {7: 18.1, 9: 18.1, 11: 30.0}[self.d_factory]

        return distillation_time_lookup() * self.factory_syndrome_extraction_cycles


@dataclass(frozen=True)
class CultivationFactory(TFactory, SurfaceCodeFactory):
    SEED: int = 0
    rng: Generator = default_rng(SEED)
    d_colour_code: int = 3
    r1: int = 3
    r2: int = 5

    @property
    @abstractmethod
    @override
    def t_prep_time(self: Self) -> float:
        def get_rounds(
            circuit_type: str,
        ) -> int:
            if circuit_type == "inject[unitary]+cultivate":
                circuit = cultiv.make_inject_and_cultivate_circuit(
                    inject_style="unitary",
                    dcolor=self.d_colour_code,
                    basis="Y",
                )
            elif circuit_type == "escape-to-big-matchable-code":
                circuit = cultiv.make_escape_to_big_matchable_code_circuit(
                    dcolor=self.d_colour_code,
                    dsurface=self.d_factory,
                    basis="Y",
                    r_growing=self.r1,
                    r_end=self.r2,
                )
            else:
                raise NotImplementedError(f"Unsupported circuit_type: {circuit_type!r}")

            return gen.count_measurement_layers(circuit)

        def quick_calc_prob(stats_path: str) -> float:
            total_shots = 0
            total_discards = 0
            with open(stats_path, newline="") as f:
                reader = DictReader(f)
                for row in reader:
                    total_shots += int(row["shots"])
                    total_discards += int(row["discards"])

            if total_shots == 0:
                raise ValueError("No shots recorded in stats CSV.")

            prob = 1 - (total_discards / total_shots)
            return prob

        def _generate_circuit(circuit_type, gateset):
            """Run make_circuits to produce the .stim file, exactly as cmd.sh does."""
            cmd = [
                str(MAKE_CIRCUITS),
                "--circuit_type",
                circuit_type,
                "--noise_strength",
                str(self.physical_qubit_error_rate),
                "--gateset",
                gateset,
                "--basis",
                "Y",
                "--d1",
                str(self.d_colour_code),
                "--out_dir",
                str(SINTER_OUTPUTS_DIR),
            ]
            if circuit_type == "escape-to-big-matchable-code":
                cmd += ["--r1", str(self.r1)]
                cmd += ["--r2", str(self.r2)]
                cmd += ["--d2", str(self.d_factory)]

            env = environ.copy()
            env["PYTHONPATH"] = str(SINTER_SRC)
            print(f"Running: PYTHONPATH={SINTER_SRC} {' '.join(cmd)}")
            result = run(cmd, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"make_circuits failed:\n{result.stderr}")
            print(result.stdout.strip())

        def _find_circuit_file(circuit_type: str, gateset: str):
            r = get_rounds(
                circuit_type=circuit_type,
            )
            noise = "uniform" if gateset == "css" else "si1000"

            if circuit_type == "inject[unitary]+cultivate":
                noiseless = cultiv.make_inject_and_cultivate_circuit(
                    inject_style="unitary", dcolor=self.d_colour_code, basis="Y"
                )
                meta = {
                    "c": circuit_type,
                    "p": self.physical_qubit_error_rate,
                    "noise": noise,
                    "g": gateset,
                    "q": noiseless.num_qubits,
                    "b": "Y",
                    "r": r,
                    "d1": self.d_colour_code,
                }
            elif circuit_type == "escape-to-big-matchable-code":
                noiseless = cultiv.make_escape_to_big_matchable_code_circuit(
                    dcolor=self.d_colour_code,
                    dsurface=self.d_factory,
                    basis="Y",
                    r_growing=self.r1,
                    r_end=self.r2,
                )
                meta = {
                    "c": circuit_type,
                    "p": self.physical_qubit_error_rate,
                    "noise": noise,
                    "g": gateset,
                    "q": noiseless.num_qubits,
                    "b": "Y",
                    "r": r,
                    "r1": self.r1,
                    "d1": self.d_colour_code,
                    "r2": self.r2,
                    "d2": self.d_factory,
                }
            else:
                raise NotImplementedError(f"Unsupported circuit_type: {circuit_type!r}")

            meta_str = ",".join(f"{k}={v}" for k, v in meta.items())
            return SINTER_OUTPUTS_DIR / f"{meta_str}.stim"

        def get_success_prob(
            circuit_type: str,
            gateset: str,
            decoder: str = "perfectionist",
            max_shots: int = 10_000_000,
            stats_file_name: str = "stats.csv",
        ) -> float:
            circuit_path = _find_circuit_file(circuit_type, gateset)

            if not circuit_path.exists():
                print("Circuit file not found, running make_circuits...")
                _generate_circuit(circuit_type, gateset)
            else:
                print(f"Using existing circuit: {circuit_path.name}")

            SINTER_OUTPUTS_DIR.mkdir(exist_ok=True)
            stats_path = SINTER_OUTPUTS_DIR / stats_file_name
            stats_path.write_text(
                "shots,errors,discards,seconds,decoder,strong_id,json_metadata\n"
            )

            print(f"Stats file: {stats_path.resolve()}")

            cmd = [
                "sinter",
                "collect",
                "--metadata_func",
                "auto",
                "--circuits",
                str(circuit_path),
                "--decoders",
                decoder,
                "--max_shots",
                str(max_shots),
                "--custom_decoders",
                "cultiv:sinter_samplers",
                "--save_resume_filepath",
                str(SINTER_OUTPUTS_DIR / stats_file_name),
            ]
            env = environ.copy()
            env["PYTHONPATH"] = str(SINTER_SRC)

            print(f"Running: PYTHONPATH={SINTER_SRC} {' '.join(cmd)}")
            result = run(cmd, env=env, capture_output=True, text=True)

            if result.returncode != 0:
                print("sinter stderr:", result.stderr)
                raise RuntimeError(
                    f"sinter collect failed with exit code {result.returncode}"
                )

            return quick_calc_prob(str(SINTER_OUTPUTS_DIR / stats_file_name))

        def stage1_2_success_probability() -> float:
            stats_file_name = (
                f"stats_inject_unitary_cultivate,d1={self.d_colour_code},"
                f"noise_strength={self.physical_qubit_error_rate}.csv"
            )
            stats_full_path = SINTER_OUTPUTS_DIR / stats_file_name
            if stats_full_path.exists():
                prob = quick_calc_prob(stats_path=str(stats_full_path))
            else:
                prob = get_success_prob(
                    circuit_type="inject[unitary]+cultivate",
                    gateset="css",
                    decoder="perfectionist",
                    stats_file_name=stats_file_name,
                )
            return prob

        def stage3_success_probability() -> float:
            stats_file_name = (
                f"stats_escape_to_big_matchable_code,d1={self.d_colour_code},"
                f"d2={self.d_factory},r1={self.r1},r2={self.r2},"
                f"noise_strength={self.physical_qubit_error_rate}.csv"
            )
            stats_full_path = SINTER_OUTPUTS_DIR / stats_file_name
            if stats_full_path.exists():
                prob = quick_calc_prob(stats_path=str(stats_full_path))
            else:
                prob = get_success_prob(
                    circuit_type="escape-to-big-matchable-code",
                    gateset="css",
                    decoder="desaturation",
                    stats_file_name=stats_file_name,
                )
            return prob

        n_color = (3 * (self.d_colour_code * self.d_colour_code) + 1) / 4

        c1 = get_rounds("inject[unitary]+cultivate")
        c2 = get_rounds("escape-to-big-matchable-code")

        p12 = stage1_2_success_probability()
        p3 = stage3_success_probability()

        def one_round() -> float:
            time = 0.0

            while True:  # escape success
                while True:  # parallel success of colour code preparation
                    time += c1
                    if log(self.rng.uniform(0.0, 1.0)) >= (
                        self.d_factory * self.d_factory
                    ) / n_color * log(1 - p12):
                        break  # parallel preparations and we loop if all fail

                time += c2
                if log(p3) >= log(self.rng.uniform(0.0, 1.0)):
                    break  # loop if escape fails

            return time

        return one_round()


@dataclass(frozen=True)
class RzFactory(Factory):
    @property
    @abstractmethod
    def t_factory(self: Self) -> TFactory | None:
        pass


@dataclass(frozen=True)
class TtoRzSurfaceCodeFactory(RzFactory, SurfaceCodeFactory):
    t_factory_type: str = "Distillation"

    @property
    @override
    def synthesis_logical_error_rate(self: Self) -> float:
        return (
            self.t_factory.num_t_injections
            * self.t_factory.synthesis_logical_error_rate
        )

    @property
    @override
    def t_factory(self: Self) -> TFactory:
        if self.t_factory_type == "Distillation":
            return DistillationFactory(
                d_factory=self.d_factory,
                physical_qubit_error_rate=self.physical_qubit_error_rate,
            )
        elif self.t_factory_type == "Cultivation":
            return CultivationFactory(
                d_factory=self.d_factory,
                physical_qubit_error_rate=self.physical_qubit_error_rate,
            )
        else:
            raise ValueError(f"Unknown T factory type: {self.t_factory_type}")


@dataclass(frozen=True)
class SuperconductingSurfaceCodeFactory(TtoRzSurfaceCodeFactory):
    @property
    @override
    def synthesis_logical_error_rate(self: Self) -> float:
        return (
            self.t_factory.num_t_injections
            * self.t_factory.synthesis_logical_error_rate
        )

    @property
    @override
    def factory_prep_time(self: Self) -> float:
        t_prep_time = self.t_factory.factory_prep_time
        return self.t_factory.num_t_injections * (
            t_prep_time + self.d_factory * self.factory_syndrome_extraction_cycles
        )


@dataclass(frozen=True)
class NeutralAtomSurfaceCodeFactory(TtoRzSurfaceCodeFactory):
    @property
    @override
    def synthesis_logical_error_rate(self: Self) -> float:
        return (
            self.t_factory.num_t_injections
            * self.t_factory.synthesis_logical_error_rate
        )

    # FIXME: check if this is the correct timing model
    @property
    @override
    def factory_prep_time(self: Self) -> float:
        t_prep_time = self.t_factory.factory_prep_time
        return self.t_factory.num_t_injections * (
            t_prep_time + self.factory_syndrome_extraction_cycles
        )


@dataclass(frozen=True)
class STARSurfaceCodeFactory(RzFactory, SurfaceCodeFactory):
    SEED: int = 0
    rng: Generator = default_rng(SEED)

    @property
    @override
    def synthesis_logical_error_rate(self: Self) -> float:
        if self.physical_qubit_error_rate != 1e-4:
            raise NotImplementedError(
                "Only physical qubit error rate of 1e-4 is supported for now."
            )

        return {
            7: 10**-7.5,
            9: 10**-9.1,
        }[self.d_factory]

    @property
    @override
    def factory_prep_time(self: Self) -> float:
        def circuit_success_probability(number_of_operations: int) -> float:
            # computes the probability that an even number of checks flip
            result = 0.0
            for i in range(0, number_of_operations, 2):
                from math import comb

                result += (
                    comb(number_of_operations, i)
                    * (self.physical_qubit_error_rate**i)
                    * (
                        (1 - self.physical_qubit_error_rate)
                        ** (number_of_operations - i)
                    )
                )
            return result

        """
        The circuit is as follows:
        GATES --- ROTATION --- ANCILLA PREPARATION --- MEASUREMENT1 --- ANCILLA PREPARATION --- MEASUREMENT2
          4   ---     1    ---         12          ---       4      ---         12          ---       4
        Upto MEASUREMENT1, we can combine all errors and the success probability is stored in `mThetaSuccessUptoMeasurement1` (= `s1`)
        Between MEASUREMENT1 and MEASUREMENT2, we can combine all errors and the success probability is stored in `mThetaSuccessBetweenMeasurement12` (= `s2`)
        The success probability of each group of MEASUREMENTs is stored in `mThetaMeasurementSuccessProbability` (= `sm`)
        The success probability of the whole circuit is stored in `mThetaSuccessProbability`
        `mThetaSuccessProbability` is the probability of getting a `1111` for both measurements under the assumption that each error flips between `1111`
            and another bit string (and vice versa) with equal probability and the error probability is `physicalQubitErrorRate` for all gates and measurements
        """

        s1 = circuit_success_probability(16)
        sm = (1 - self.physical_qubit_error_rate) ** 4
        s2 = circuit_success_probability(12)
        m_theta_success_probability = s1 * sm * (s2 * sm + (1 - s2) * (1 - sm) / 15) + (
            1 - s1
        ) * (1 - sm) / 15 * (
            (1 - s2) / 15 * sm + (s2 + 14 * (1 - s2) / 15) * (1 - sm) / 15
        )
        log_expansion_success_probability = (
            (self.d_factory * self.d_factory - 1) / 2
        ) * (log(circuit_success_probability(7)) + log(circuit_success_probability(5)))

        def one_round() -> float:
            time = 0.0

            while True:  # expansion success
                while True:  # parallel success of [[4,1,1,2]] preparation
                    time += 4.0
                    if log(self.rng.uniform(0.0, 1.0)) >= (
                        (self.d_factory - 1) / 2
                    ) ** 2 * log(1 - m_theta_success_probability):
                        break  # ((d - 1) / 2)^2 parallel preparations and we loop if all fail

                time += 2 * self.factory_syndrome_extraction_cycles
                if log_expansion_success_probability >= log(self.rng.uniform(0.0, 1.0)):
                    break  # loop if expansion fails

            return time

        return one_round()


@dataclass(frozen=True)
class ColourCodeFactory(RzFactory):
    @property
    @abstractmethod
    @override
    def factory_prep_time(self: Self) -> float:
        pass  # TODO: implement this and error rates


@dataclass(frozen=True)
class TimingModel:
    factory: Factory

    d_gross: int = 10
    idle_gross_code: int = 8

    in_module_step: float = 12 * d_gross
    AVERAGE_IN_MODULE_COUNTS: float = 18.5
    inter_module_layer_step: float = 12 * d_gross

    @property
    @abstractmethod
    def t_prep_time(self: Self) -> float:
        pass

    @abstractmethod
    def rz_injection_time(
        self: Self,
        aware_time: float,
        ready_time: float,
        angle: float,
    ) -> tuple[float, int]:
        pass


@dataclass(frozen=True)
class TDGTimingModel(TimingModel):
    factory: TFactory

    @override
    def rz_injection_time(
        self: Self,
        aware_time: float,
        ready_time: float,
        angle: float,
    ) -> tuple[float, int]:
        return ready_time + self.factory.num_t_injections * (
            self.factory.factory_prep_time + self.inter_module_layer_step
        ), self.factory.num_t_injections


@dataclass(frozen=True)
class INJEQTTimingModel(TimingModel):
    factory: RzFactory
    num_factories: int = 1
    factory_availabilities: dict[int, tuple[float | None, float]] = field(init=False)
    SEED: int = 0
    rng: Generator = default_rng(SEED)

    def __post_init__(self):
        initial_dict = {i: (None, 0) for i in range(self.num_factories)}

        object.__setattr__(self, "factory_availabilities", initial_dict)

    def _consume_factory(
        self: Self, angle: float, aware_time: float, ready_time: float
    ) -> float:
        """
        Consumes the factory at the given index,
        updates the factory availability time,
        and returns the time at which the injection can start.
        """
        factory_availabilities = self.factory_availabilities
        earliest_factory_index, earliest_consumption_time = -1, float("inf")
        next_angle = 2 * angle
        for factory_index, (
            factory_angle,
            available_time,
        ) in factory_availabilities.items():
            next_angle = max(
                next_angle, 2 * factory_angle if factory_angle is not None else 0
            )
            if (
                factory_angle is None
                or abs(fmod(factory_angle - angle, 2 * pi)) < 1e-10
            ) and available_time < earliest_consumption_time:
                earliest_factory_index = factory_index
                earliest_consumption_time = available_time

        factory_prep_time = self.factory.factory_prep_time
        if earliest_factory_index == -1:
            for factory_index in range(self.num_factories):
                factory_availabilities[factory_index] = (
                    None,
                    aware_time + factory_prep_time,
                )
            return self._consume_factory(angle, aware_time, ready_time)

        factory_availabilities[earliest_factory_index] = (
            next_angle,
            earliest_consumption_time + factory_prep_time,
        )
        return max(earliest_consumption_time, ready_time)

    @override
    def rz_injection_time(
        self: Self,
        aware_time: float,
        ready_time: float,
        angle: float,
    ) -> tuple[float, int]:
        success = False
        num_injections = 0
        while not success:
            ready_time = (
                self._consume_factory(angle, aware_time, ready_time)
                + self.inter_module_layer_step
            )
            num_injections += 1
            success = self.rng.random() < 0.5
            angle *= 2
        return ready_time, num_injections
