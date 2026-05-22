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

from oqd_analog_emulator.conversion import (
    QutipBackendCompiler,
    QutipExperimentVM,
    QutipMetricConversion,
)
from oqd_analog_emulator.datastore import task_result_to_datastore

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
    Run a [`QutipExperiment`][oqd_analog_emulator.interface.QutipExperiment] and return a dataschema [`Datastore`][oqd_dataschema.datastore.Datastore].

    Args:
        model (QutipExperiment):
        args: Compiled QuTiP task arguments.

    Returns:
        Datastore containing an [`AnalogEmulatorDataGroup`][oqd_dataschema.groups.analog_emulator.AnalogEmulatorDataGroup] under the ``emulation`` key.

    """
    n_qreg = model.n_qreg
    n_qmode = model.n_qmode
    metrics = Post(QutipMetricConversion(n_qreg=n_qreg, n_qmode=n_qmode))(args.metrics)
    vm = QutipExperimentVM(
        qt_metrics=metrics,
        n_shots=args.n_shots,
        fock_cutoff=args.fock_cutoff,
        dt=args.dt,
    )
    interpreter = Pre(vm)
    interpreter(model=model)

    return task_result_to_datastore(
        vm.results,
        args,
        state_trajectory=vm._state_trajectory,
        measurements=vm._measurements,
    )
