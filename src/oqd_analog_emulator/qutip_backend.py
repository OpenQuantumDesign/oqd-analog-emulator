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
from oqd_core.backend.program import AnalogProgram
from oqd_core.analysis.analog.cfg import AnalogCFGBuilder
from oqd_core.analysis.analog.type_checker import AnalogTypeChecker
from oqd_core.frontend.analog import parse_analog
from oqd_core.analysis.analog.symbol_table import AnalogSymbolTableBuilder
from oqd_core.compiler.analog.passes.compile import compile_analog_circuit
from oqd_analog_emulator.interpreter import QutipInterpreter

from typing import Optional

########################################################################################

__all__ = [
    "QutipBackend",
]

########################################################################################


class QutipBackend(BackendBase):
    """
    Class representing the Qutip backend
    """
    
    def compile(self, source):
        circuit = parse_analog(source)
        cfg = AnalogCFGBuilder().run(circuit)
        checker = AnalogTypeChecker(cfg)

        symbol_analysis = AnalogSymbolTableBuilder(cfg, checker.dataflow_result)
        symbol_table = symbol_analysis.symbol_table
        
        circuit, cfg = compile_analog_circuit(circuit=circuit, cfg=cfg, symbol_table=symbol_table)
        
        program = AnalogProgram(circuit=circuit, cfg=cfg, symbol_table=symbol_table)

        return program
        

    def run(
        self,
        source: Optional[str] = "",
        program: Optional[AnalogProgram] = None,
    ):
        """
        Method to simulate an experiment using the QuTip backend

        Args:
            source (Optional[str]): Run experiment from a valid analog code.
            program (Optional[AnalogProgram]): Run experiment from a valid AnalogProgram object.
        Returns:
            Program and Interpreter objects.

        Note:
            only one of source or program must be provided.
        """
        
        if source:
            program = self.compile(source)
        
        if not program:
            raise ValueError("Provide valid analog code or AnalogProgram.")
            
        cfg = program.cfg

        interpreter = QutipInterpreter(graph=cfg) 
        interpreter.run()
        
        return program, interpreter

