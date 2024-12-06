# Copyright 2024 Open Quantum Design

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

import numpy as np


########################################################################################


from oqd_core.interface.analog.operator import *
from oqd_core.interface.analog.operation import *
from oqd_core.interface.math import MathStr
from oqd_core.backend.metric import *
from oqd_core.backend.task import Task, TaskArgsAnalog
from oqd_analog_emulator.qutip_backend import QutipBackend

########################################################################################

X, Y, Z, I, A, C, LI = (
    PauliX(),
    PauliY(),
    PauliZ(),
    PauliI(),
    Annihilation(),
    Creation(),
    Identity(),
)


@pytest.fixture
def one_qubit_rabi_flopping_protocol():
    Hx = AnalogGate(hamiltonian=-(np.pi / 4) * X)
    ac = AnalogCircuit()
    ac.evolve(duration=1, gate=Hx)
    ac.evolve(duration=1, gate=Hx)
    ac.evolve(duration=1, gate=Hx)
    # define task args
    args = TaskArgsAnalog(
        n_shots=100,
        fock_cutoff=4,
        metrics={
            "Z": Expectation(operator=Z),
        },
        dt=1e-3,
    )
    return ac, args


@pytest.fixture
def bell_state_standard_protocol():

    Hii = AnalogGate(hamiltonian=I @ I)
    Hxi = AnalogGate(hamiltonian=X @ I)
    Hyi = AnalogGate(hamiltonian=Y @ I)
    Hxx = AnalogGate(hamiltonian=X @ X)
    Hmix = AnalogGate(hamiltonian=(-1) * (I @ X))
    Hmxi = AnalogGate(hamiltonian=(-1) * (X @ I))
    Hmyi = AnalogGate(hamiltonian=(-1) * (Y @ I))

    ac = AnalogCircuit()

    # Hadamard
    ac.evolve(duration=(3 * np.pi) / 2, gate=Hii)
    ac.evolve(duration=np.pi / 2, gate=Hxi)
    ac.evolve(duration=np.pi / 4, gate=Hmyi)

    # CNOT
    ac.evolve(duration=np.pi / 4, gate=Hyi)
    ac.evolve(duration=np.pi / 4, gate=Hxx)
    ac.evolve(duration=np.pi / 4, gate=Hmix)
    ac.evolve(duration=np.pi / 4, gate=Hmxi)
    ac.evolve(duration=np.pi / 4, gate=Hmyi)
    ac.evolve(duration=np.pi / 4, gate=Hii)

    # define task args
    args = TaskArgsAnalog(
        n_shots=100,
        fock_cutoff=4,
        metrics={
            "Z^0": Expectation(operator=Z @ I),
            "Z^1": Expectation(operator=I @ Z),
        },
        dt=1e-2,
    )

    return ac, args


@pytest.fixture
def three_qubit_GHz_protocol():

    # Hadamard on first qubit
    Hii = AnalogGate(hamiltonian=I @ I @ I)
    Hxi = AnalogGate(hamiltonian=X @ I @ I)
    Hyi = AnalogGate(hamiltonian=Y @ I @ I)

    # CNOT on Second
    Hxx2 = AnalogGate(hamiltonian=X @ X @ I)
    Hmix2 = AnalogGate(hamiltonian=(-1) * (I @ X @ I))

    Hmxi = AnalogGate(hamiltonian=(-1) * (X @ I @ I))
    Hmyi = AnalogGate(hamiltonian=(-1) * (Y @ I @ I))

    # CNOT on Third
    Hxx3 = AnalogGate(hamiltonian=X @ I @ X)
    Hmix3 = AnalogGate(hamiltonian=(-1) * (I @ I @ X))
    ac = AnalogCircuit()

    # Hadamard
    ac.evolve(duration=(3 * np.pi) / 2, gate=Hii)
    ac.evolve(duration=np.pi / 2, gate=Hxi)
    ac.evolve(duration=np.pi / 4, gate=Hmyi)

    # CNOT
    ac.evolve(duration=np.pi / 4, gate=Hyi)
    ac.evolve(duration=np.pi / 4, gate=Hxx2)
    ac.evolve(duration=np.pi / 4, gate=Hmix2)
    ac.evolve(duration=np.pi / 4, gate=Hmxi)
    ac.evolve(duration=np.pi / 4, gate=Hmyi)
    ac.evolve(duration=np.pi / 4, gate=Hii)

    # CNOT
    ac.evolve(duration=np.pi / 4, gate=Hyi)
    ac.evolve(duration=np.pi / 4, gate=Hxx3)
    ac.evolve(duration=np.pi / 4, gate=Hmix3)
    ac.evolve(duration=np.pi / 4, gate=Hmxi)
    ac.evolve(duration=np.pi / 4, gate=Hmyi)
    ac.evolve(duration=np.pi / 4, gate=Hii)

    # define task args
    args = TaskArgsAnalog(
        n_shots=500,
        fock_cutoff=4,
        metrics={
            "Z^0": Expectation(operator=Z @ I @ I),
            "Z^1": Expectation(operator=I @ Z @ I),
            "Z^2": Expectation(operator=I @ I @ Z),
        },
        dt=1e-2,
    )

    return ac, args


def get_amplitude_arrays(state: list):
    real_amplitudes, imag_amplitudes = [], []
    for x in state:
        real_amplitudes.append(x.real)
        imag_amplitudes.append(x.imag)
    return real_amplitudes, imag_amplitudes


def assert_lists_close(list1, list2, tolerance=0.001):
    assert len(list1) == len(list2), "The input lists have different length"
    for i, (elem1, elem2) in enumerate(zip(list1, list2)):
        assert abs(elem1 - elem2) <= tolerance, "List elements {i}, {elem1} and {elem2}, are out of tolerance"



def test_one_qubit_rabi_flopping(one_qubit_rabi_flopping_protocol):
    """One qubit rabi flopping"""

    ac, args = one_qubit_rabi_flopping_protocol

    task = Task(program=ac, args=args)

    backend = QutipBackend()

    results = backend.run(task=task)

    real_amplitudes, imag_amplitudes = get_amplitude_arrays(results.state)

    assert_lists_close(real_amplitudes, [-0.707, 0])
    assert_lists_close(imag_amplitudes, [0, 0.707])
    assert abs(results.metrics["Z"][-1] - 0) <= 0.001


def test_bell_state_standard(bell_state_standard_protocol):
    """Standard Bell State preparation"""

    ac, args = bell_state_standard_protocol

    task = Task(program=ac, args=args)

    backend = QutipBackend()

    results = backend.run(task=task)

    real_amplitudes, imag_amplitudes = get_amplitude_arrays(results.state)

    assert_lists_close(real_amplitudes, [0.707, 0, 0, 0.707])
    assert_lists_close(imag_amplitudes, [0, 0, 0, 0])
    assert abs(results.metrics["Z^0"][-1] - 0) <= 0.001
    assert abs(results.metrics["Z^1"][-1] - 0) <= 0.001


def test_ghz_state(three_qubit_GHz_protocol):
    """Standard GHz State preparation"""

    ac, args = three_qubit_GHz_protocol

    task = Task(program=ac, args=args)

    backend = QutipBackend()

    results = backend.run(task=task)

    real_amplitudes, imag_amplitudes = get_amplitude_arrays(results.state)

    assert_lists_close(real_amplitudes, [0.707, 0, 0, 0, 0, 0, 0, 0.707])
    assert_lists_close(imag_amplitudes, [0, 0, 0, 0, 0, 0, 0, 0])
    assert abs(results.metrics["Z^0"][-1] - 0) <= 0.001
    assert abs(results.metrics["Z^1"][-1] - 0) <= 0.001
    assert abs(results.metrics["Z^2"][-1] - 0) <= 0.001


def test_identity_operation_simple():
    """Simple Identity operation using inverse"""
    H1 = AnalogGate(hamiltonian=(-1) * X)
    H1_inv = AnalogGate(hamiltonian=X)

    ac = AnalogCircuit()
    ac.evolve(duration=1, gate=H1)
    ac.evolve(duration=1, gate=H1_inv)
    # define task args
    args = TaskArgsAnalog(
        n_shots=100,
        fock_cutoff=4,
        metrics={
            "Z": Expectation(operator=Z),
        },
        dt=1e-3,
    )
    task = Task(program=ac, args=args)

    backend = QutipBackend()

    results = backend.run(task=task)

    real_amplitudes, imag_amplitudes = get_amplitude_arrays(results.state)

    assert_lists_close(real_amplitudes, [1, 0])
    assert_lists_close(imag_amplitudes, [0, 0])
    assert abs(results.metrics["Z"][-1] - 1) <= 0.001


def test_identity_operation_nested():
    """Nested Identity operation using inverse"""
    H1 = AnalogGate(hamiltonian=(-1) * X)
    H1_inv = AnalogGate(hamiltonian=X)

    ac = AnalogCircuit()
    ac.evolve(duration=1, gate=H1)
    ac.evolve(duration=1, gate=H1_inv)
    ac.evolve(duration=1, gate=H1)
    ac.evolve(duration=1, gate=H1_inv)
    ac.evolve(duration=1, gate=H1)
    ac.evolve(duration=1, gate=H1_inv)
    # define task args
    args = TaskArgsAnalog(
        n_shots=100,
        fock_cutoff=4,
        metrics={
            "Z": Expectation(operator=Z),
        },
        dt=1e-3,
    )
    task = Task(program=ac, args=args)

    backend = QutipBackend()

    results = backend.run(task=task)

    real_amplitudes, imag_amplitudes = get_amplitude_arrays(results.state)

    assert_lists_close(real_amplitudes, [1, 0])
    assert_lists_close(imag_amplitudes, [0, 0])
    assert abs(results.metrics["Z"][-1] - 1) <= 0.001


def test_identity_operation_three_qubit_simple():
    """Simple Identity operation using inverse for 3 qubits"""
    H1 = AnalogGate(hamiltonian=(-1) * (X @ Y @ Z))
    H1_inv = AnalogGate(hamiltonian=X @ Y @ Z)

    ac = AnalogCircuit()
    ac.evolve(duration=1, gate=H1)
    ac.evolve(duration=1, gate=H1_inv)
    # define task args
    args = TaskArgsAnalog(
        n_shots=500,
        fock_cutoff=4,
        metrics={
            "Z^0": Expectation(operator=Z @ I @ I),
            "Z^1": Expectation(operator=I @ Z @ I),
            "Z^2": Expectation(operator=I @ I @ Z),
        },
        dt=1e-2,
    )

    task = Task(program=ac, args=args)

    backend = QutipBackend()

    results = backend.run(task=task)

    real_amplitudes, imag_amplitudes = get_amplitude_arrays(results.state)

    assert_lists_close(real_amplitudes, [1, 0, 0, 0, 0, 0, 0, 0])
    assert_lists_close(imag_amplitudes, [0, 0, 0, 0, 0, 0, 0, 0])
    assert abs(results.metrics["Z^0"][-1] - 1) <= 0.001
    assert abs(results.metrics["Z^1"][-1] - 1) <= 0.001
    assert abs(results.metrics["Z^2"][-1] - 1) <= 0.001

def test_identity_operation_three_qubit_nested():
    """Nested Identity operation using inverse for 3 qubits"""
    H1 = AnalogGate(hamiltonian=(-1) * (X @ Y @ Z))
    H1_inv = AnalogGate(hamiltonian=X @ Y @ Z)

    H2 = AnalogGate(hamiltonian=(-1) * (X @ X @ X))
    H2_inv = AnalogGate(hamiltonian=X @ X @ X)

    H3 = AnalogGate(hamiltonian=(-1) * (I @ X @ I))
    H3_inv = AnalogGate(hamiltonian=I @ X @ I)

    ac = AnalogCircuit()
    ac.evolve(duration=1, gate=H1)
    ac.evolve(duration=1, gate=H1_inv)
    ac.evolve(duration=1, gate=H2)
    ac.evolve(duration=1, gate=H2_inv)
    ac.evolve(duration=1, gate=H3)
    ac.evolve(duration=1, gate=H3_inv)

    # define task args
    args = TaskArgsAnalog(
        n_shots=500,
        fock_cutoff=4,
        metrics={
            "Z^0": Expectation(operator=Z @ I @ I),
            "Z^1": Expectation(operator=I @ Z @ I),
            "Z^2": Expectation(operator=I @ I @ Z),
        },
        dt=1e-2,
    )

    task = Task(program=ac, args=args)

    backend = QutipBackend()

    results = backend.run(task=task)

    real_amplitudes, imag_amplitudes = get_amplitude_arrays(results.state)

    assert_lists_close(real_amplitudes, [1, 0, 0, 0, 0, 0, 0, 0])
    assert_lists_close(imag_amplitudes, [0, 0, 0, 0, 0, 0, 0, 0])
    assert abs(results.metrics["Z^0"][-1] - 1) <= 0.001
    assert abs(results.metrics["Z^1"][-1] - 1) <= 0.001
    assert abs(results.metrics["Z^2"][-1] - 1) <= 0.001


def test_metrics_count_none(one_qubit_rabi_flopping_protocol):
    """Testing without metrics and counts"""

    ac, _ = one_qubit_rabi_flopping_protocol

    # define task args
    args = TaskArgsAnalog(
        n_shots=None,
        fock_cutoff=4,
        metrics={},
        dt=1e-3,
    )

    task = Task(program=ac, args=args)

    backend = QutipBackend()

    results = backend.run(task=task)

    assert results.counts == {}
    assert results.metrics == {}



def test_one_qubit_rabi_flopping(one_qubit_rabi_flopping_protocol):
    """One qubit rabi flopping canonicalization"""

    _, args = one_qubit_rabi_flopping_protocol

    Hx = AnalogGate(hamiltonian=-(np.pi / 8) * (2 * X))

    ac = AnalogCircuit()
    ac.evolve(duration=1, gate=Hx)
    ac.evolve(duration=1, gate=Hx)
    ac.evolve(duration=1, gate=Hx)

    task = Task(program=ac, args=args)

    backend = QutipBackend()

    results = backend.run(task=task)

    real_amplitudes, imag_amplitudes = get_amplitude_arrays(results.state)

    assert np.allclose(real_amplitudes, [-0.707, 0], atol=0.001)
    assert np.allclose(imag_amplitudes, [0, 0.707], atol=0.001)
    assert pytest.approx(results.metrics["Z"][-1], abs=0.001) == 0


def test_bell_state_standard(bell_state_standard_protocol):
    """Standard Bell State preparation canonicalization"""

    _, args = bell_state_standard_protocol

    Hii = AnalogGate(hamiltonian=1 * (I @ I))
    Hxi = AnalogGate(hamiltonian=(X @ I))  # Scalar Multiplication not given
    Hyi = AnalogGate(hamiltonian=1 * (Y @ I))
    Hxx = AnalogGate(
        hamiltonian=1 * (X @ (I * X * I))
    )  # multiplication by identity
    Hmix = AnalogGate(hamiltonian=(-1) * (I @ X))
    Hmxi = AnalogGate(hamiltonian=(-1) * (X @ I))
    Hmyi = AnalogGate(hamiltonian=(-0.5) * (Y @ (2 * I)))  # scalar multiplication

    ac = AnalogCircuit()

    # Hadamard
    ac.evolve(duration=(3 * np.pi) / 2, gate=Hii)
    ac.evolve(duration=np.pi / 2, gate=Hxi)
    ac.evolve(duration=np.pi / 4, gate=Hmyi)

    # CNOT
    ac.evolve(duration=np.pi / 4, gate=Hyi)
    ac.evolve(duration=np.pi / 4, gate=Hxx)
    ac.evolve(duration=np.pi / 4, gate=Hmix)
    ac.evolve(duration=np.pi / 4, gate=Hmxi)
    ac.evolve(duration=np.pi / 4, gate=Hmyi)
    ac.evolve(duration=np.pi / 4, gate=Hii)

    task = Task(program=ac, args=args)

    backend = QutipBackend()

    results = backend.run(task=task)

    real_amplitudes, imag_amplitudes = get_amplitude_arrays(results.state)

    assert np.allclose(real_amplitudes, [0.707, 0, 0, 0.707], atol=0.001)
    assert np.allclose(imag_amplitudes, [0, 0, 0, 0], atol=0.001)
    assert pytest.approx(results.metrics["Z^0"][-1], abs=0.001) == 0
    assert pytest.approx(results.metrics["Z^1"][-1], abs=0.001) == 0
