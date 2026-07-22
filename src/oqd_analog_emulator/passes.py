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

from oqd_analog_emulator.rewrite import (
    QutipBackendCompiler,
    QutipMetricConversion,
)

from oqd_compiler_infrastructure import Post, Pre
from oqd_core.interface.analog.circuit import AnalogCircuit
########################################################################################

__all__ = [
    # "compiler_analog_circuit_to_qutipIR",
    # "compiler_analog_args_to_qutipIR",
    "compiler_analog_circuit_to_qutip_backend",
    # "run_qutip_experiment",
]

########################################################################################


def compiler_analog_circuit_to_qutip_backend(model: AnalogCircuit, args, n_qreg, n_qmode):
    """
    This compiles ([`AnalogCircuit`][oqd_core.interface.analog.operation.AnalogCircuit] to a list of  [`QutipOperation`][oqd_analog_emulator.interface.QutipOperation] objects

    Args:
        model (AnalogCircuit):
        fock_cutoff (int): fock_cutoff for Ladder Operators

    Returns:
        (list(QutipOperation)):

    """
    metrics = Post(QutipMetricConversion(n_qreg=n_qreg, n_qmode=n_qmode))(args.metrics)

    interpreter = Pre(QutipBackendCompiler(
            qt_metrics=metrics,
            n_shots=args.n_shots,
            fock_cutoff=args.fock_cutoff,
            dt=args.dt,
            n_qreg=n_qreg,
            n_qmode=n_qmode,
        )
    )
    interpreter(model=model)

    return interpreter.children[0].results

