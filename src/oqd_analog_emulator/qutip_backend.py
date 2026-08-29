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

from oqd_core.analysis.analog.cfg import AnalogCFGBuilder
from oqd_core.analysis.analog.symbol_table import AnalogSymbolTableBuilder
from oqd_core.analysis.analog.type_checker import AnalogTypeChecker
from oqd_core.backend.base import BackendBase
from oqd_core.backend.program import AnalogProgram
from oqd_core.compiler.analog.passes.compile import compile_analog_circuit
from oqd_core.frontend.analog import parse_analog

from oqd_analog_emulator.interpreter import QutipInterpreter

########################################################################################

__all__ = [
    "QutipBackend",
]

########################################################################################


class QutipBackend(BackendBase):
    """
    Class representing the Qutip backend
    """

    def compile(self, program: str):
        circuit = parse_analog(program)
        cfg = AnalogCFGBuilder().run(circuit)
        checker = AnalogTypeChecker(cfg)

        symbol_analysis = AnalogSymbolTableBuilder(cfg, checker.dataflow_result)
        symbol_table = symbol_analysis.symbol_table

        circuit, cfg = compile_analog_circuit(
            circuit=circuit, cfg=cfg, symbol_table=symbol_table
        )

        program = AnalogProgram(circuit=circuit, cfg=cfg, symbol_table=symbol_table)

        return program

    def run(self, program: str | AnalogProgram = None, n_shots: int = 10, fock_cutoff: int = 4, dt: float = 0.1,):
        """
        Method to simulate an experiment using the QuTip backend

        Args:
            program (str | AnalogProgram): Run experiment from valid analog code or AnalogProgram object.
        Returns:
            Program object, Interpreter object, and the output of the QuTip simulation.
        """

        if isinstance(program, str):
            program = self.compile(program)

        if not isinstance(program, AnalogProgram):
            raise TypeError("Provide valid analog code or AnalogProgram.")

        cfg = program.cfg

        interpreter = QutipInterpreter(graph=cfg, n_shots=n_shots, fock_cutoff=fock_cutoff, dt=dt)
        output = interpreter.run()

        return program, interpreter, output
