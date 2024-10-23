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

from oqd_analog_emulator.conversion import (
    QutipBackendCompiler,
    QutipExperimentVM,
    QutipMetricConversion,
)

from oqd_compiler_infrastructure import Post, Pre

########################################################################################

__all__ = [
    "compiler_analog_circuit_to_qutipIR",
    "compiler_analog_args_to_qutipIR",
    "run_qutip_experiment",
]

########################################################################################


def compiler_analog_circuit_to_qutipIR(model, fock_cutoff):
    """
    This compiles ([`AnalogCircuit`][oqd_core.interface.analog.operation.AnalogCircuit] to a list of  [`QutipOperation`][oqd_analog_emulator.interface.QutipOperation] objects

    Args:
        model (AnalogCircuit):
        fock_cutoff (int): fock_cutoff for Ladder Operators

    Returns:
        (list(QutipOperation)):

    """
    return Post(QutipBackendCompiler(fock_cutoff=fock_cutoff))(model=model)


def compiler_analog_args_to_qutipIR(model):
    """
    This compiles TaskArgsAnalog to a list of TaskArgsQutip


    Args:
        model (TaskArgsAnalog):

    Returns:
        (TaskArgsQutip):

    """
    return Post(QutipBackendCompiler(fock_cutoff=model.fock_cutoff))(model=model)


def run_qutip_experiment(model: QutipExperimentVM, args):
    """
    This takes in a [`QutipExperiment`][oqd_analog_emulator.interface.QutipExperiment] and produces a TaskResultAnalog object

    Args:
        model (QutipExperiment):
        args: (Qutip

    Returns:
        (TaskResultAnalog): Contains results of the simulation

    """
    n_qreg = model.n_qreg
    n_qmode = model.n_qmode
    metrics = Post(QutipMetricConversion(n_qreg=n_qreg, n_qmode=n_qmode))(args.metrics)
    interpreter = Pre(
        QutipExperimentVM(
            qt_metrics=metrics,
            n_shots=args.n_shots,
            fock_cutoff=args.fock_cutoff,
            dt=args.dt,
        )
    )
    interpreter(model=model)

    return interpreter.children[0].results
