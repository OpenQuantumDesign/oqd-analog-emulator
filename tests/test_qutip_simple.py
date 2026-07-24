# Copyright 2024-2025 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from oqd_core.analysis.analog.cfg import AnalogCFGBuilder
from oqd_core.analysis.analog.symbol_table import AnalogSymbolTableBuilder
from oqd_core.analysis.analog.type_checker import AnalogTypeChecker
from oqd_core.backend.metric import Expectation
from oqd_core.backend.program import AnalogProgram
from oqd_core.backend.task import Task, TaskArgsAnalog
from oqd_core.frontend.analog import parse_analog
from oqd_core.interface.analog.expr import (
    Annihilation,
    Creation,
    Identity,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
)

from oqd_analog_emulator.qutip_backend import QutipBackend


@pytest.fixture
def get_program():
    # printer = Post(PrettyPrint())
    
    source = "r = qreg(1) \n H = -(3.14 / 4) %* %X \n  evolve(H, 1, r)"

    circuit = parse_analog(source)
    cfg = AnalogCFGBuilder().run(circuit)
    checker = AnalogTypeChecker(cfg)
    symbol_analysis = AnalogSymbolTableBuilder(cfg, checker.dataflow_result)
    symbol_table = symbol_analysis.symbol_table

    program = AnalogProgram(circuit=circuit, cfg=cfg, symbol_table=symbol_table)
    return program

X, Y, Z, PI, A, C, LI = (
    PauliX(),
    PauliY(),
    PauliZ(),
    PauliI(),
    Annihilation(),
    Creation(),
    Identity(),
)


def get_amplitude_arrays(state: list):
    real_amplitudes, imag_amplitudes = [], []
    for x in state:
        real_amplitudes.append(x.real)
        imag_amplitudes.append(x.imag)
    return real_amplitudes, imag_amplitudes


def assert_lists_close(list1, list2, tolerance=0.001):
    assert len(list1) == len(list2), "The input lists have different length"
    for i, (elem1, elem2) in enumerate(zip(list1, list2)):
        assert (
            abs(elem1 - elem2) <= tolerance
        ), "List elements {i}, {elem1} and {elem2}, are out of tolerance"


def test_one_qubit_rabi_flopping(get_program):
    """One qubit rabi flopping"""
    args = TaskArgsAnalog(
        n_shots=100,
        fock_cutoff=4,
        metrics={
            "Z": Expectation(operator=Z),
        },
        dt=1e-3,
    )

    task = Task(program=get_program, args=args)

    backend = QutipBackend()

    results = backend.run(task=task)
    print(results)

    # real_amplitudes, imag_amplitudes = get_amplitude_arrays(results.state)

    # assert_lists_close(real_amplitudes, [-0.707, 0])
    # assert_lists_close(imag_amplitudes, [0, 0.707])
    # assert abs(results.metrics["Z"][-1] - 0) <= 0.001
    

