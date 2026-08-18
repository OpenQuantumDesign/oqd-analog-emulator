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

from oqd_compiler_infrastructure import Post, Pre
from oqd_core.interface.analog.circuit import AnalogCircuit
import qutip as qt
import numpy as np
from oqd_compiler_infrastructure import ConversionRule
from oqd_analog_emulator.rewrite import (
    QutipBackendCompiler,
    QutipMetricConversion,
)
from oqd_dataschema import Datastore, GroupBase, Dataset

########################################################################################


class QutipBackendCompiler(ConversionRule):
    """
    This is a ConversionRule which compiles analog layer objects to QutipExperiment objects

    Args:
        model (VisitableBaseModel): This takes in objects in Analog level and converts them to representations which can be used to run QuTip simulations.

    Returns:
        model (Union[VisitableBaseModel, Any]): QuTip objects and representations which can be used to run QuTip simulations

    """

    def __init__(self, fock_cutoff=None, current_time=None):
        super().__init__()
        self._fock_cutoff = fock_cutoff
        self.current_time = current_time

    def map_PauliI(self, model, operands):
        op = qt.qeye(2)
        return qt.QobjEvo(op)

    def map_PauliX(self, model, operands):
        op = qt.sigmax()
        return qt.QobjEvo(op)

    def map_PauliY(self, model, operands):
        op = qt.sigmay()
        return qt.QobjEvo(op)

    def map_PauliZ(self, model, operands):
        op = qt.sigmaz()
        return qt.QobjEvo(op)

    def map_Identity(self, model, operands):
        op = qt.qeye(self._fock_cutoff)
        return qt.QobjEvo(op)

    def map_Annihilation(self, model, operands):
        op = qt.destroy(self._fock_cutoff)
        return qt.QobjEvo(op)

    def map_Creation(self, model, operands):
        op = qt.create(self._fock_cutoff)
        return qt.QobjEvo(op)

    def map_OperatorAdd(self, model, operands):
        return operands["op1"] + operands["op2"]

    def map_OperatorMul(self, model, operands):
        # print(operands)
        return qt.QobjEvo(lambda t: operands["op1"](t) * operands["op2"](t))

    def map_OperatorKron(self, model, operands):
        return qt.tensor(operands["op1"], operands["op2"])

    def map_MathNum(self, model, operands):
        return lambda t: model.value

    def map_MathImag(self, model, operands):
        return lambda t: 1j

    def map_MathVar(self, model, operands):
        if model.name == "#t":
            return lambda t: t
        
        if model.name == "#s":
            return lambda t: t - self.current_time

        raise ValueError(
            f"Unsupported variable {model.name}, only variable t is supported"
        )

    def map_MathFunc(self, model, operands):
        if model.func in [
            "abs",
            "sin",
            "cos",
            "tan",
            "exp",
            "log",
            "sinh",
            "cosh",
            "tanh",
            "atan",
            "acos",
            "asin",
            "atanh",
            "asinh",
            "acosh",
            "conj",
            "real",
            "imag",
            "atan2",
        ]:
            if isinstance(operands["expr"], list):
                return lambda t: getattr(np, model.func)(
                    *[o(t) for o in operands["expr"]]
                )
            return lambda t: getattr(np, model.func)(operands["expr"](t))

        if model.func == "heaviside":
            return lambda t: np.heaviside(operands["expr"](t), 1)

        raise ValueError(f"Unsupported function {model.func}")

    def map_MathAdd(self, model, operands):
        return lambda t: operands["expr1"](t) + operands["expr2"](t)

    def map_MathSub(self, model, operands):
        return lambda t: operands["expr1"](t) - operands["expr2"](t)

    def map_MathMul(self, model, operands):
        return lambda t: operands["expr1"](t) * operands["expr2"](t)

    def map_MathDiv(self, model, operands):
        return lambda t: operands["expr1"](t) / operands["expr2"](t)

    def map_MathPow(self, model, operands):
        return lambda t: operands["expr1"](t) ** operands["expr2"](t)


