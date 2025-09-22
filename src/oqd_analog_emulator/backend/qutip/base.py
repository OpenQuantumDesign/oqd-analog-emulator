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

from typing import Annotated, Dict, List, Union
from oqd_compiler_infrastructure import Post, Pre, TypeReflectBaseModel
from oqd_core.interface.analog import AnalogCircuit, OperatorSubTypes
from oqd_core.backend import BackendBase

from pydantic import Discriminator, NonNegativeInt

from oqd_analog_emulator.backend.qutip.conversion import (
    QutipExperimentVM,
    QutipMetricConversion,
)
from oqd_analog_emulator.backend.qutip.passes import (
    compiler_analog_args_to_qutipIR,
    compiler_analog_circuit_to_qutipIR,
)

from oqd_core.compiler.analog.passes.canonicalize import (
    analog_operator_canonicalization,
)
from oqd_core.compiler.analog.passes.assign import (
    assign_analog_circuit_dim,
)

########################################################################################

__all__ = [
    "QutipBackend",
    "Expectation",
    "EntanglementEntropyVN",
    "EntanglementEntropyRenyi",
]

########################################################################################


class Metric(TypeReflectBaseModel):
    pass


class Expectation(Metric):
    operator: OperatorSubTypes


class EntanglementEntropyVN(Metric):
    qreg: List[NonNegativeInt] = []
    qmode: List[NonNegativeInt] = []


class EntanglementEntropyRenyi(Metric):
    alpha: NonNegativeInt = 1
    qreg: List[NonNegativeInt] = []
    qmode: List[NonNegativeInt] = []


MetricSubTypes = Annotated[
    Union[
        Expectation,
        EntanglementEntropyVN,
        EntanglementEntropyRenyi,
    ],
    Discriminator(discriminator="class_"),
]


class QutipBackendArgs(TypeReflectBaseModel):
    n_shots: Union[int, None] = 10
    fock_cutoff: int = 4
    dt: float = 0.1
    metrics: Dict[str, MetricSubTypes] = {}


########################################################################################


class QutipBackend(BackendBase):
    """
    Class representing the Qutip backend
    """

    def compile(self, program: AnalogCircuit, args: QutipBackendArgs):
        """
        Method for compiling program of task to a [`QutipExperiment`][oqd_analog_emulator.interface.QutipExperiment] and converting
        args of task to [`TaskArgsAnalog`][oqd_core.backend.task.TaskArgsAnalog].

        Args:
            task (Task): Quantum experiment to compile

        Returns:
            converted_circuit (QutipExperiment): QutipExperiment containing the compiled experiment for Qutip
            converted_args (TaskArgsQutip): args of analog layer are converted to args for QuTip.

        """
        # pass to canonicaliza the operators in the AnalogCircuit
        canonicalized_circuit = analog_operator_canonicalization(program)

        # This just canonicalizes the operators inside the TaskArgsAnalog
        # i.e. operators for Expectation
        canonicalized_args = analog_operator_canonicalization(args)

        # another pass which assigns the n_qreg and n_qmode of the
        # AnalogCircuit IR
        assigned_circuit = assign_analog_circuit_dim(canonicalized_circuit)

        # # This just verifies that the operators in the args have the same
        # # dimension as the operators in the AnalogCircuit
        # verify_analog_args_dim(
        #     canonicalized_args,
        #     n_qreg=assigned_circuit.n_qreg,
        #     n_qmode=assigned_circuit.n_qmode,
        # )

        # another pass which compiles AnalogCircuit to a QutipExperiment
        converted_circuit = compiler_analog_circuit_to_qutipIR(
            assigned_circuit, fock_cutoff=args.fock_cutoff
        )

        # This just converts the args so that the operators of the args are
        # converted to qutip objects
        converted_args = compiler_analog_args_to_qutipIR(canonicalized_args)

        return (
            converted_circuit,
            converted_args,
        )

    def run(self, program: AnalogCircuit, args: QutipBackendArgs):
        """
        Method to simulate an experiment using the QuTip backend

        Args:
            task (Task): Run experiment from a [`Task`][oqd_core.backend.task.Task] object

        Returns:
            TaskResultAnalog object containing the simulation results.

        """

        # if experiment is None and args is not None:
        #     raise TypeError("args provided without QuTip experiment")
        # if experiment is not None and args is None:
        #     raise TypeError("QuTip experiment provided without args")
        #
        # if task is not None and experiment is not None:
        #     raise TypeError("Both task and experiment are given as inputs to run")
        # if experiment is None:
        #     experiment, args = self.compile(task=task)

        experiment, args = self.compile(program=program, args=args)

        n_qreg = experiment.n_qreg
        n_qmode = experiment.n_qmode
        metrics = Post(QutipMetricConversion(n_qreg=n_qreg, n_qmode=n_qmode))(
            args["metrics"]
        )
        interpreter = Pre(
            QutipExperimentVM(
                qt_metrics=metrics,
                n_shots=args["n_shots"],
                fock_cutoff=args["fock_cutoff"],
                dt=args["dt"],
            )
        )
        interpreter(experiment)

        return interpreter.children[0].results
