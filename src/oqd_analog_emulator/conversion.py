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

import itertools
import time

import numpy as np
import qutip as qt

from oqd_compiler_infrastructure import ConversionRule, RewriteRule, Chain
from oqd_core.compiler.math.passes import (
    evaluate_math_expr,
    simplify_math_expr,
    print_math_expr,
)

########################################################################################

from oqd_analog_emulator.interface import (
    QutipExperiment,
    QutipOperation,
    QutipMeasurement,
    TaskArgsQutip,
    QutipExpectation,
)

########################################################################################

__all__ = [
    "entanglement_entropy_vn",
    "QutipMetricConversion",
    "QutipBackendCompiler",
    "QutipExperimentVM",
]


def _qobj_to_state_vector(state: qt.Qobj) -> np.ndarray:
    """Extract a 1D complex128 state vector from a QuTiP Qobj."""
    return np.asarray(state.full(), dtype=np.complex128).squeeze()


########################################################################################


def entanglement_entropy_vn(t, psi, qreg, qmode, n_qreg, n_qmode):
    rho = qt.ptrace(
        psi,
        qreg + [n_qreg + m for m in qmode],
    )
    return qt.entropy_vn(rho)


class QutipMetricConversion(ConversionRule):
    """
    This takes in a a dictionary containing Metrics, which get converted to lambda functions for QuTip

    Args:
        model (dict): The values are Analog layer Operators

    Returns:
        model (dict): The values are lambda functions

    Note:
        n_qreg and n_qmode are given as compiler parameters
    """

    def __init__(self, n_qreg, n_qmode):
        super().__init__()
        self._n_qreg = n_qreg
        self._n_qmode = n_qmode

    def map_QutipExpectation(self, model, operands):
        assert len(model.operator) > 0, "List of operator terms must be non-empty"

        op_exp = None
        for idx, operator in enumerate(model.operator):
            coefficient = evaluate_math_expr(operator[1])
            if idx == 0:
                op_exp = coefficient * operator[0]
            else:
                op_exp + coefficient * operator[0]

        return lambda t, psi: qt.expect(op_exp, psi)

    def map_EntanglementEntropyVN(self, model, operands):
        return lambda t, psi: entanglement_entropy_vn(
            t, psi, model.qreg, model.qmode, self._n_qreg, self._n_qmode
        )


class QutipExperimentVM(RewriteRule):
    """
    Virtual machine that simulates a [`QutipExperiment`][oqd_analog_emulator.interface.QutipExperiment]
    and collects run data as plain numpy-friendly attributes for datastore export.

    Attributes populated during execution:
        times: simulation time points
        metric_labels: ordered metric names
        metrics: expectation time-series per metric
        state_trajectory: state vectors at each time step
        measurements: sampled qubit outcomes after a measure instruction
        runtime: total solver wall time in seconds
    """

    def __init__(self, qt_metrics, n_shots, fock_cutoff, dt):
        super().__init__()
        self._qt_metrics = qt_metrics
        self._n_shots = n_shots
        self._fock_cutoff = fock_cutoff
        self._dt = dt

        self.metric_labels: list[str] = list(qt_metrics.keys())
        self.times: list[float] = []
        self.metrics: dict[str, list[float]] = {key: [] for key in self.metric_labels}
        self.state_trajectory: list[np.ndarray] = []
        self.measurements: np.ndarray | None = None
        self.runtime: float = 0.0

    def map_QutipExperiment(self, model):
        dims = model.n_qreg * [2] + model.n_qmode * [self._fock_cutoff]
        self.n_qreg = model.n_qreg
        self.n_qmode = model.n_qmode
        self.current_state = qt.tensor([qt.basis(d, 0) for d in dims])

        self.times.append(0.0)
        self.state_trajectory.append(_qobj_to_state_vector(self.current_state))
        for key in self.metric_labels:
            self.metrics[key].append(self._qt_metrics[key](0.0, self.current_state))

    def map_QutipMeasurement(self, model):
        if self._n_shots is not None:
            probs = np.power(np.abs(self.current_state.full()), 2).squeeze()
            inds = np.random.choice(len(probs), size=self._n_shots, p=probs)
            opts = self.n_qreg * [[0, 1]] + self.n_qmode * [
                list(range(self._fock_cutoff))
            ]
            bases = list(itertools.product(*opts))
            shots = np.array([bases[ind] for ind in inds])
            self.measurements = shots[:, : self.n_qreg].astype(np.int64)

    def map_QutipOperation(self, model):
        duration = model.duration
        tspan = np.linspace(0, duration, round(duration / self._dt)).tolist()

        qutip_hamiltonian = []
        for op, coeff in model.hamiltonian:
            qutip_hamiltonian.append(
                [op, Chain(simplify_math_expr, print_math_expr)(coeff)]
            )

        start_runtime = time.time()
        result_qobj = qt.sesolve(
            qutip_hamiltonian,
            self.current_state,
            tspan,
            e_ops=self._qt_metrics,
            options={"store_states": True},
        )
        self.runtime += time.time() - start_runtime

        self.times.extend([t + self.times[-1] for t in tspan][1:])

        for idx, key in enumerate(self.metric_labels):
            self.metrics[key].extend(result_qobj.expect[idx].tolist()[1:])

        if result_qobj.states is not None:
            for st in result_qobj.states[1:]:
                self.state_trajectory.append(_qobj_to_state_vector(st))

        self.current_state = result_qobj.final_state


class QutipBackendCompiler(ConversionRule):
    """
    This is a ConversionRule which compiles analog layer objects to QutipExperiment objects

    Args:
        model (VisitableBaseModel): This takes in objects in Analog level and converts them to representations which can be used to run QuTip simulations.

    Returns:
        model (Union[VisitableBaseModel, Any]): QuTip objects and representations which can be used to run QuTip simulations

    """

    def __init__(self, fock_cutoff=None):
        super().__init__()
        self._fock_cutoff = fock_cutoff

    def map_AnalogCircuit(self, model, operands):
        return QutipExperiment(
            instructions=operands["sequence"],
            n_qreg=operands["n_qreg"],
            n_qmode=operands["n_qmode"],
        )

    def map_TaskArgsAnalog(self, model, operands):
        return TaskArgsQutip(
            layer=model.layer,
            n_shots=model.n_shots,
            fock_cutoff=model.fock_cutoff,
            dt=model.dt,
            metrics=operands["metrics"],
        )

    def map_Expectation(self, model, operands):
        return QutipExpectation(operator=operands["operator"])

    def map_Evolve(self, model, operands):
        return QutipOperation(
            hamiltonian=operands["gate"],
            duration=model.duration,
        )

    def map_Measure(self, model, operands):
        return QutipMeasurement()

    def map_AnalogGate(self, model, operands):
        return operands["hamiltonian"]

    def map_OperatorAdd(self, model, operands):
        op = operands["op1"]
        op.append(operands["op2"][0])
        return op

    def map_OperatorScalarMul(self, model, operands):
        return [(operands["op"], model.expr)]

    def map_PauliI(self, model, operands):
        return qt.qeye(2)

    def map_PauliX(self, model, operands):
        return qt.sigmax()

    def map_PauliY(self, model, operands):
        return qt.sigmay()

    def map_PauliZ(self, model, operands):
        return qt.sigmaz()

    def map_Identity(self, model, operands):
        return qt.qeye(self._fock_cutoff)

    def map_Creation(self, model, operands):
        return qt.create(self._fock_cutoff)

    def map_Annihilation(self, model, operands):
        return qt.destroy(self._fock_cutoff)

    def map_OperatorMul(self, model, operands):
        return operands["op1"] * operands["op2"]

    def map_OperatorKron(self, model, operands):
        return qt.tensor(operands["op1"], operands["op2"])
