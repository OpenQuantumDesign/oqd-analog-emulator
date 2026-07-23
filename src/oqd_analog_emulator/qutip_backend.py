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

from oqd_core.backend.base import BackendBase
from oqd_core.backend.task import Task

from oqd_analog_emulator.passes import (
    # run_qutip_experiment,
    compiler_analog_circuit_to_qutip_backend,
)

from oqd_core.compiler.analog.passes.compile import compile_analog_circuit


########################################################################################

__all__ = [
    "QutipBackend",
]

########################################################################################


class QutipBackend(BackendBase):
    """
    Class representing the Qutip backend
    """ 

    def run(
        self,
        task: Task,
    ):
        """
        Method to simulate an experiment using the QuTip backend

        Args:
            task (Optional[Task]): Run experiment from a [`Task`][oqd_core.backend.task.Task] object

        Returns:
            TaskResultAnalog object containing the simulation results.

        Note:
            only one of task or experiment must be provided.
        """
        
        circuit = task.program.circuit
        cfg = task.program.cfg
        symbol_table = task.program.symbol_table
        
        circuit, cfg = compile_analog_circuit(circuit=circuit, cfg=cfg, symbol_table=symbol_table)

        # another pass which compiles AnalogCircuit to a QutipExperiment
        return compiler_analog_circuit_to_qutip_backend(model=circuit, args=task.args)

