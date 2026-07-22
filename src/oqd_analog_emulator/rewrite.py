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
import itertools
import time
import qutip as qt

from oqd_compiler_infrastructure import RewriteRule, Chain
from oqd_core.backend.task import TaskResultAnalog
from oqd_core.backend.metric import EntanglementEntropyVN, Expectation
from oqd_core.interface.analog.expr import (
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
    Identity,
    Annihilation,
    Creation,
    OperatorMul,
    OperatorKron,
    OperatorAdd,
    MathMul,
    Evolve,
    Measure,
)
from oqd_core.interface.analog.circuit import AnalogCircuit
from oqd_core.compiler.analog.math.passes import (
    evaluate_math_expr,
    simplify_math_expr,
    print_math_expr,
)

def entanglement_entropy_vn(t, psi, qreg, qmode, n_qreg, n_qmode):
    rho = qt.ptrace(
        psi,
        qreg + [n_qreg + m for m in qmode],
    )
    return qt.entropy_vn(rho)

class QutipMetricConversion(RewriteRule):
    """
    This takes in a dictionary containing Metrics, which get converted to lambda functions for QuTip

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
    
    def map_Expectation(self, model: Expectation):
        assert len(model.operator) > 0, "List of operator terms must be non-empty"

        op_exp = None
        for idx, operator in enumerate(model.operator):
            coefficient = evaluate_math_expr(operator[1])
            if idx == 0:
                op_exp = coefficient * operator[0]
            else:
                op_exp + coefficient * operator[0]

        return lambda t, psi: qt.expect(op_exp, psi)

    def map_EntanglementEntropyVN(self, model: EntanglementEntropyVN):
        return lambda t, psi: entanglement_entropy_vn(
            t, psi, model.qreg, model.qmode, self._n_qreg, self._n_qmode
        )


class QutipBackendCompiler(RewriteRule):
    def __init__(self, qt_metrics, n_shots, fock_cutoff, dt, n_qreg, n_qmode):
        super().__init__()
        self.results = TaskResultAnalog(runtime=0)
        self._qt_metrics = qt_metrics
        self._n_shots = n_shots
        self._fock_cutoff = fock_cutoff
        self._dt = dt
        self.n_qreg=n_qreg
        self.n_qmode=n_qmode
    
    def map_PauliI(self, model: PauliI):
        return qt.qeye(2)
    
    def map_PauliX(self, model: PauliX):
        return qt.sigmax()

    def map_PauliY(self, model: PauliY):
        return qt.sigmay()

    def map_PauliZ(self, model: PauliZ):
        return qt.sigmaz()

    def map_Identity(self, model: Identity):
        return qt.qeye(self._fock_cutoff)

    def map_Creation(self, model: Annihilation):
        return qt.create(self._fock_cutoff)

    def map_Annihilation(self, model: Creation):
        return qt.destroy(self._fock_cutoff)

    def map_OperatorMul(self, model: OperatorMul):
        return model.op1 * model.op2

    def map_OperatorKron(self, model: OperatorKron):
        return qt.tensor(model.op1, model.op2)
    
    def map_OperatorAdd(self, model: OperatorAdd):
        return [model.op1, model.op2]
    
    def map_MathMul(self, model: MathMul):
        return model.expr1 * model.expr2
    
    def map_Evolve(self, model: Evolve):
        # return QutipOperation(
        #     hamiltonian=model.hamiltonian,
        #     duration=model.duration,
        # )
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
        self.results.runtime = time.time() - start_runtime + self.results.runtime

        self.results.times.extend([t + self.results.times[-1] for t in tspan][1:])

        for idx, key in enumerate(self.results.metrics.keys()):
            self.results.metrics[key].extend(result_qobj.expect[idx].tolist()[1:])

        self.current_state = result_qobj.final_state

        self.results.state = list(
            result_qobj.final_state.full().squeeze(),
        )

    def map_Measure(self, model: Measure):
        # return QutipMeasurement()
        if self._n_shots is None:
            self.results.counts = {}
        else:
            probs = np.power(np.abs(self.current_state.full()), 2).squeeze()
            n_shots = self._n_shots
            inds = np.random.choice(len(probs), size=n_shots, p=probs)
            opts = self.n_qreg * [[0, 1]] + self.n_qmode * [
                list(range(self._fock_cutoff))
            ]
            bases = list(itertools.product(*opts))
            shots = np.array([bases[ind] for ind in inds])
            bitstrings = ["".join(map(str, shot)) for shot in shots]
            self.results.counts = {
                bitstring: bitstrings.count(bitstring) for bitstring in bitstrings
            }

        self.results.state = list(
            self.current_state.full().squeeze(),
        )
    
    def map_AnalogCircuit(self, model: AnalogCircuit):
        # return QutipExperiment(
        #     instructions=model.statements
        # )
        dims = self.n_qreg * [2] + self.n_qmode * [self._fock_cutoff]
        self.current_state = qt.tensor([qt.basis(d, 0) for d in dims])

        self.results.times.append(0.0)
        self.results.state = list(
            self.current_state.full().squeeze(),
        )
        self.results.metrics.update(
            {
                key: [self._qt_metrics[key](0.0, self.current_state)]
                for key in self._qt_metrics.keys()
            }
        )

    