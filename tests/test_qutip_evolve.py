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

import numpy as np
import pytest
import qutip
from oqd_core.compiler.analog.error import AnalogCompilerError

from oqd_analog_emulator.method_table import QuantumRegister, RegisterName
from oqd_analog_emulator.qutip_backend import QutipBackend

########################################################################################


TICK = "\033[1;32m\u2714\033[0m"
CROSS = "\033[1;31m\u2718\033[0m"


def assert_register_eq(register, expected):
    if register.__class__ != expected.__class__:
        raise TypeError(
            f"register class [{register.__class__.__name__}] does not match expected [{expected.__class__.__name__}]"
        )

    expected = expected.sort()

    m_name = register.name == expected.name
    m_time = np.isclose(register.time, expected.time, atol=1e-4)
    m_time_last_updated = np.isclose(
        register.time_last_updated, expected.time_last_updated, atol=1e-4
    )
    m_state = np.isclose(register.state.full(), expected.state.full(), atol=1e-4).all()

    if all((m_name, m_time_last_updated, m_state)):
        return

    print(f"\n{' diff ':=^80}")

    if not m_name:
        print(f"{' name ':-^80}")
        print(f"Got: {register.name}")
        print(f"Expected: {expected.name}")

    if not m_time:
        print(f"{' time ':-^80}")
        print(f"Got: {register.time}")
        print(f"Expected: {expected.time}")

    if not m_time_last_updated:
        print(f"{' time last updated ':-^80}")
        print(f"Got: {register.time_last_updated}")
        print(f"Expected: {expected.time_last_updated}")

    if not m_state:
        print(f"{' state ':-^80}")
        print(f"Got: {register.state}")
        print(f"Expected: {expected.state}")

    print("=" * 80)

    raise ValueError(
        "register does not match expected "
        "\033[0m["
        f"name {TICK if m_name else CROSS}, "
        f"time {TICK if m_time else CROSS}, "
        f"time_last_updated {TICK if m_time_last_updated else CROSS}, "
        f"state {TICK if m_state else CROSS}"
        "]"
    )


def prepare_initial_state(state, register):
    H = " %+ ".join(
        [
            " %@ ".join(
                [
                    "%Y" if i == n else "%I"
                    for i in range(len(register))
                    if state[i] == 1
                ]
            )
            for n, sigma in enumerate(state)
            if sigma == 1
        ]
    )

    if not H:
        return ""

    return f"evolve({H}, pi/2, [{','.join([r for n, r in enumerate(register) if state[n] == 1])}])"


########################################################################################


@pytest.fixture
def backend():

    return QutipBackend()


########################################################################################


@pytest.mark.xfail(
    raises=AnalogCompilerError,
    reason="Hamiltonian does not match targets dimension",
)
def test_xfail_incompatible_dim(backend):
    source = """
    q = qreg(2)
    initialize(q)

    evolve(%Y , 1, q)

    q
    """

    program, interp, out = backend.run(source)


@pytest.mark.xfail(
    raises=AnalogCompilerError,
    reason="Hamiltonian acts on mode register does not match targets a qubit register",
)
def test_xfail_incorrect_register_type(backend):
    source = """
    q = qreg(2)
    initialize(q)

    evolve(%A , 1, q)

    q
    """

    program, interp, out = backend.run(source)


########################################################################################


class TestQutipQubitsEvolve:
    def test_qutip_init(self, backend):
        source = """
        q = qreg(2)
        initialize(q)

        q
        """

        program, interp, out = backend.run(source)

        assert_register_eq(
            out[0][1],
            QuantumRegister(
                name=[RegisterName(name="q", index=0, dim=2)],
                time=0,
                time_last_updated=0,
                state=qutip.basis(2, 0),
            ),
        )

        assert_register_eq(
            out[1][1],
            QuantumRegister(
                name=[RegisterName(name="q", index=1, dim=2)],
                time=0,
                time_last_updated=0,
                state=qutip.basis(2, 0),
            ),
        )

    def test_qutip_reorder(self, backend):
        source = """
        q = qreg(2)
        initialize(q)

        [q[1],q[0]]
        """

        program, interp, out = backend.run(source)

        assert_register_eq(
            out[0][1],
            QuantumRegister(
                name=[RegisterName(name="q", index=1, dim=2)],
                time=0,
                time_last_updated=0,
                state=qutip.basis(2, 0),
            ),
        )

        assert_register_eq(
            out[1][1],
            QuantumRegister(
                name=[RegisterName(name="q", index=0, dim=2)],
                time=0,
                time_last_updated=0,
                state=qutip.basis(2, 0),
            ),
        )

    def test_qutip_multi_register_init(self, backend):
        source = """
        q = qreg(1)
        r = qreg(1)
        initialize([q[0], r[0]])

        [q[0],r[0]]
        """

        program, interp, out = backend.run(source)

        assert_register_eq(
            out[0][1],
            QuantumRegister(
                name=[RegisterName(name="q", index=0, dim=2)],
                time=0,
                time_last_updated=0,
                state=qutip.basis(2, 0),
            ),
        )

        assert_register_eq(
            out[1][1],
            QuantumRegister(
                name=[RegisterName(name="r", index=0, dim=2)],
                time=0,
                time_last_updated=0,
                state=qutip.basis(2, 0),
            ),
        )

    @pytest.mark.parametrize(
        "stmt",
        [
            "evolve(%X, 1, q[0])",
            "evolve(%X %@ %X, 1, [q[0], q[2]])",
            "evolve(%X %@ %X %@ %X, 1, q)",
        ],
    )
    def test_qutip_evolve_output(self, stmt, backend):
        source = f"""
        q = qreg(3)
        initialize(q)

        {stmt}
        """

        program, interp, out = backend.run(source)

        assert out == []

    def test_qutip_evolve(self, backend):
        source = """
        q = qreg(1)
        initialize(q)

        evolve(%X, 1, q[0])

        q
        """

        program, interp, out = backend.run(source)

        assert_register_eq(
            out[0][1],
            QuantumRegister(
                name=[RegisterName(name="q", index=0, dim=2)],
                time=1,
                time_last_updated=1,
                state=np.cos(1) * qutip.basis(2, 0)
                - 1j * np.sin(1) * qutip.basis(2, 1),
            ),
        )

    def test_qutip_evolve_time(self, backend):
        source = """
        q = qreg(2)
        initialize(q)

        evolve(%I, 10, q[0])
        evolve(%X, 1, q[1])

        q
        """

        program, interp, out = backend.run(source)

        assert_register_eq(
            out[0][1],
            QuantumRegister(
                name=[RegisterName(name="q", index=0, dim=2)],
                time=11,
                time_last_updated=10,
                state=np.exp(-1j * 10) * qutip.basis(2, 0),
            ),
        )

        assert_register_eq(
            out[1][1],
            QuantumRegister(
                name=[RegisterName(name="q", index=1, dim=2)],
                time=11,
                time_last_updated=11,
                state=np.cos(1) * qutip.basis(2, 0)
                - 1j * np.sin(1) * qutip.basis(2, 1),
            ),
        )

    @pytest.mark.skip(reason="TODO: fix behavior of reinitializing a qubit")
    def test_qutip_reinit(self, backend):
        source = """
        q = qreg(1)
        initialize(q)

        evolve(%X, 1, q[0])

        initialize(q)

        [q[0]]
        """

        program, interp, out = backend.run(source)

        assert_register_eq(
            out[0][1],
            QuantumRegister(
                name=[RegisterName(name="q", index=0, dim=2)],
                time_last_updated=1,
                state=qutip.basis(2, 0),
            ),
        )

    ########################################################################################

    @pytest.mark.parametrize("phi", np.logspace(-3, 1, 11))
    def test_qutip_ry(self, phi, backend):
        source = f"""
        q = qreg(1)
        initialize(q)

        evolve(%Y, {phi}, q[0])

        q
        """

        program, interp, out = backend.run(source)

        assert_register_eq(
            out[0][1],
            QuantumRegister(
                name=[RegisterName(name="q", index=0, dim=2)],
                time=phi,
                time_last_updated=phi,
                state=np.cos(phi) * qutip.basis(2, 0) + np.sin(phi) * qutip.basis(2, 1),
            ),
        )

    @pytest.mark.parametrize("phi", np.logspace(-3, 1, 11))
    def test_qutip_rxx(self, phi, backend):
        source = f"""
        q = qreg(2)
        initialize(q)

        evolve(%X %@ %X, {phi}, q)

        q
        """

        program, interp, out = backend.run(source)

        assert_register_eq(out[0][1], out[1][1])

        assert_register_eq(
            out[0][1],
            QuantumRegister(
                name=[
                    RegisterName(name="q", index=0, dim=2),
                    RegisterName(name="q", index=1, dim=2),
                ],
                time=phi,
                time_last_updated=phi,
                state=np.cos(phi) * qutip.basis([2, 2], [0, 0])
                - 1j * np.sin(phi) * qutip.basis([2, 2], [1, 1]),
            ),
        )

    ########################################################################################

    @pytest.mark.parametrize(
        ("initial_state", "expected"),
        [
            (
                [0, 0],
                QuantumRegister(
                    name=[
                        RegisterName(name="q", index=0, dim=2),
                        RegisterName(name="q", index=1, dim=2),
                    ],
                    time=np.pi / 4,
                    time_last_updated=np.pi / 4,
                    state=(
                        qutip.basis([2, 2], [0, 0]) - 1j * qutip.basis([2, 2], [1, 1])
                    )
                    / np.sqrt(2),
                ),
            ),
            (
                [0, 1],
                QuantumRegister(
                    name=[
                        RegisterName(name="q", index=0, dim=2),
                        RegisterName(name="q", index=1, dim=2),
                    ],
                    time=np.pi * 3 / 4,
                    time_last_updated=np.pi * 3 / 4,
                    state=(
                        qutip.basis([2, 2], [0, 1]) - 1j * qutip.basis([2, 2], [1, 0])
                    )
                    / np.sqrt(2),
                ),
            ),
            (
                [1, 0],
                QuantumRegister(
                    name=[
                        RegisterName(name="q", index=0, dim=2),
                        RegisterName(name="q", index=1, dim=2),
                    ],
                    time=np.pi * 3 / 4,
                    time_last_updated=np.pi * 3 / 4,
                    state=(
                        qutip.basis([2, 2], [1, 0]) - 1j * qutip.basis([2, 2], [0, 1])
                    )
                    / np.sqrt(2),
                ),
            ),
            (
                [1, 1],
                QuantumRegister(
                    name=[
                        RegisterName(name="q", index=0, dim=2),
                        RegisterName(name="q", index=1, dim=2),
                    ],
                    time=np.pi * 3 / 4,
                    time_last_updated=np.pi * 3 / 4,
                    state=(
                        qutip.basis([2, 2], [1, 1]) - 1j * qutip.basis([2, 2], [0, 0])
                    )
                    / np.sqrt(2),
                ),
            ),
        ],
    )
    def test_qutip_ms(self, initial_state, expected, backend):
        source = f"""
        q = qreg(2)
        initialize(q)

        pi = 3.1415926535897932

        {prepare_initial_state(initial_state, ["q[0]", "q[1]"])}

        evolve(%X %@ %X, pi/4, q)

        q
        """

        program, interp, out = backend.run(source)

        assert_register_eq(out[0][1], out[1][1])
        assert_register_eq(out[0][1], expected)

    # ########################################################################################

    @pytest.mark.parametrize(
        ("initial_state", "expected"),
        [
            (
                [0, 0],
                QuantumRegister(
                    name=[
                        RegisterName(name="q", index=0, dim=2),
                        RegisterName(name="q", index=1, dim=2),
                    ],
                    time=np.pi * 6 / 4,
                    time_last_updated=np.pi * 6 / 4,
                    state=qutip.basis([2, 2], [0, 0]),
                ),
            ),
            (
                [0, 1],
                QuantumRegister(
                    name=[
                        RegisterName(name="q", index=0, dim=2),
                        RegisterName(name="q", index=1, dim=2),
                    ],
                    time=np.pi * 8 / 4,
                    time_last_updated=np.pi * 8 / 4,
                    state=qutip.basis([2, 2], [0, 1]),
                ),
            ),
            (
                [1, 0],
                QuantumRegister(
                    name=[
                        RegisterName(name="q", index=0, dim=2),
                        RegisterName(name="q", index=1, dim=2),
                    ],
                    time=np.pi * 8 / 4,
                    time_last_updated=np.pi * 8 / 4,
                    state=qutip.basis([2, 2], [1, 1]),
                ),
            ),
            (
                [1, 1],
                QuantumRegister(
                    name=[
                        RegisterName(name="q", index=0, dim=2),
                        RegisterName(name="q", index=1, dim=2),
                    ],
                    time=np.pi * 8 / 4,
                    time_last_updated=np.pi * 8 / 4,
                    state=qutip.basis([2, 2], [1, 0]),
                ),
            ),
        ],
    )
    def test_qutip_cnot(self, initial_state, expected, backend):
        source = f"""
        q = qreg(2)
        initialize(q)

        pi = 3.1415926535897932

        {prepare_initial_state(initial_state, ["q[0]", "q[1]"])}

        evolve(%Y, pi/4, q[0])
        evolve(%X %@ %X, pi/4, q)
        evolve(-1 %* %X, pi/4, q[1])
        evolve(-1 %* %X, pi/4, q[0])
        evolve(-1 %* %Y, pi/4, q[0])
        evolve(%I %@ %I, pi/4, q)

        q
        """

        program, interp, out = backend.run(source)

        assert_register_eq(out[0][1], out[1][1])
        assert_register_eq(out[0][1], expected)

    ########################################################################################

    @pytest.mark.parametrize("duration", np.logspace(-3, 1, 11))
    def test_qutip_global_time_dep(self, duration, backend):
        source = f"""
        q = qreg(1)
        initialize(q)

        evolve(#t %* %Y, {duration}, q[0])

        q
        """

        program, interp, out = backend.run(source)

        assert_register_eq(
            out[0][1],
            QuantumRegister(
                name=[RegisterName(name="q", index=0, dim=2)],
                time=duration,
                time_last_updated=duration,
                state=np.cos(duration**2 / 2) * qutip.basis(2, 0)
                + np.sin(duration**2 / 2) * qutip.basis(2, 1),
            ),
        )

    @pytest.mark.parametrize("duration", np.logspace(-3, 1, 11))
    def test_qutip_relative_time_dep(self, duration, backend):
        source = f"""
        q = qreg(1)
        initialize(q)

        pi = 3.1415926535897932
        
        evolve(%I, 2*pi, q[0])
        evolve(#s %* %Y, {duration}, q[0])

        q
        """

        program, interp, out = backend.run(source)

        assert_register_eq(
            out[0][1],
            QuantumRegister(
                name=[RegisterName(name="q", index=0, dim=2)],
                time=duration + 2 * np.pi,
                time_last_updated=duration + 2 * np.pi,
                state=np.cos(duration**2 / 2) * qutip.basis(2, 0)
                + np.sin(duration**2 / 2) * qutip.basis(2, 1),
            ),
        )


class TestQutipModeEvolve:
    def test_qutip_mode_init(self, backend):
        source = """
        m = qmode(1)
        initialize(m)

        m
        """

        program, interp, out = backend.run(source)

        assert_register_eq(
            out[0][1],
            QuantumRegister(
                name=[RegisterName(name="m", index=0, dim=4)],
                time=0,
                time_last_updated=0,
                state=qutip.basis(4, 0),
            ),
        )

    def test_qutip_mode_evolve_simple(self, backend):
        source = """
        m = qmode(1)
        initialize(m)

        evolve(%A %+ %C, 1, m)

        m
        """

        program, interp, out = backend.run(source, fock_cutoff=2)

        assert_register_eq(
            out[0][1],
            QuantumRegister(
                name=[RegisterName(name="m", index=0, dim=2)],
                time=1,
                time_last_updated=1,
                state=np.cos(1) * qutip.basis(2, 0)
                - 1j * np.sin(1) * qutip.basis(2, 1),
            ),
        )

    @pytest.mark.parametrize("fock_cutoff", range(2, 11))
    def test_qutip_mode_evolve_fock_cutoff(self, fock_cutoff, backend):
        source = """
        m = qmode(1)
        initialize(m)

        evolve(%A %+ %C, 1, m)

        m
        """

        program, interp, out = backend.run(source, fock_cutoff=fock_cutoff)

        assert_register_eq(
            out[0][1],
            QuantumRegister(
                name=[RegisterName(name="m", index=0, dim=fock_cutoff)],
                time=1,
                time_last_updated=1,
                state=qutip.Qobj(
                    (-1j * (qutip.create(fock_cutoff) + qutip.destroy(fock_cutoff)))
                    .expm()
                    .full()[:, 0:1]
                ),
            ),
        )


class TestQutipCombinedEvolve:
    def test_qutip_red_sideband(self, backend):
        source = """
        q = qreg(1)
        m = qmode(1)
        
        initialize(q)
        initialize(m)
        
        evolve((0.5 %* %X %+ (0.5 * 1j) %* %Y) %@ %A %+ (0.5 %* %X %- (0.5 * 1j) %* %Y) %@ %C, 1, [q[0], m[0]])

        m
        """

        program, interp, out = backend.run(source, fock_cutoff=4)

        assert_register_eq(
            out[0][1],
            QuantumRegister(
                name=[
                    RegisterName(name="m", index=0, dim=4),
                    RegisterName(name="q", index=0, dim=2),
                ],
                time=1,
                time_last_updated=1,
                state=qutip.Qobj(
                    (
                        -1j
                        * (
                            qutip.tensor(qutip.create(4), qutip.sigmam())
                            + qutip.tensor(qutip.destroy(4), qutip.sigmap())
                        )
                    )
                    .expm()
                    .full()[:, 0:1],
                    dims=[[4, 2], [1, 1]],
                ),
            ),
        )

    def test_qutip_blue_sideband(self, backend):
        source = """
        q = qreg(1)
        m = qmode(1)
        
        initialize(q)
        initialize(m)
        
        evolve((0.5 %* %X %- (0.5 * 1j) %* %Y) %@ %A %+ (0.5 %* %X %+ (0.5 * 1j) %* %Y) %@ %C, 1, [q[0], m[0]])

        m
        """

        program, interp, out = backend.run(source, fock_cutoff=4)

        assert_register_eq(
            out[0][1],
            QuantumRegister(
                name=[
                    RegisterName(name="m", index=0, dim=4),
                    RegisterName(name="q", index=0, dim=2),
                ],
                time=1,
                time_last_updated=1,
                state=qutip.Qobj(
                    (
                        -1j
                        * (
                            qutip.tensor(qutip.create(4), qutip.sigmap())
                            + qutip.tensor(qutip.destroy(4), qutip.sigmam())
                        )
                    )
                    .expm()
                    .full()[:, 0:1],
                    dims=[[4, 2], [1, 1]],
                ),
            ),
        )

    @pytest.mark.skip(
        reason="Verify hamiltonian dimension consistency not working for variables"
    )
    def test_qutip_operator_with_variable(self, backend):
        source = """
        q = qreg(1)
        m = qmode(1)
        
        initialize(q)
        initialize(m)

        sigmap = (0.5 %* %X %+ (0.5 * 1j) %* %Y)
        sigmam = (0.5 %* %X %- (0.5 * 1j) %* %Y)
        
        evolve(sigmap %@ %A %+ sigmam %@ %C, 1, [q[0], m[0]])

        m
        """

        program, interp, out = backend.run(source, fock_cutoff=4)

        assert_register_eq(
            out[0][1],
            QuantumRegister(
                name=[
                    RegisterName(name="m", index=0, dim=4),
                    RegisterName(name="q", index=0, dim=2),
                ],
                time=1,
                time_last_updated=1,
                state=qutip.Qobj(
                    (
                        -1j
                        * (
                            qutip.tensor(qutip.create(4), qutip.sigmap())
                            + qutip.tensor(qutip.destroy(4), qutip.sigmam())
                        )
                    )
                    .expm()
                    .full()[:, 0:1],
                    dims=[[4, 2], [1, 1]],
                ),
            ),
        )
