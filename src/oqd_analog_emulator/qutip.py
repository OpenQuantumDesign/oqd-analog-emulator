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

########################################################################################

from typing import Any, Dict

from oqd_compiler_infrastructure import CFG, CFGBlockAccumulator, RelabelCFGBlocks
from oqd_core.analysis.analog import AnalogCFGBuilder
from oqd_core.analysis.analog.cfg import AnalogCFGBuilder
from oqd_core.analysis.analog.dim_checker import DimensionChecker
from oqd_core.analysis.analog.reaching_def import AvailableVariableAnalysis
from oqd_core.analysis.analog.type_checker import AnalogTypeChecker
from oqd_core.backend import BackendBase
from oqd_core.frontend.analog import parse_analog
from oqd_core.interface.analog import AnalogCircuit
from pydantic import Field

from oqd_analog_emulator.interpreter import AnalogInterpreter
from oqd_analog_emulator.method_table import (
    ArithmeticMixin,
    BoolMixin,
    FunctionMixin,
    MethodTableBase,
    MethodTableOptionsBase,
    PrintMixin,
    QutipMixin,
    StackStoreMixin,
)

########################################################################################

__all__ = [
    # "QutipBackend",
    "QutipMethodTable",
]

########################################################################################


class QutipMethodTableOptions(MethodTableOptionsBase):
    fock_cutoff: int = 4
    dt: float = 1e-3
    singleshot_init: bool = False
    ignore_measurements: bool = False
    ignore_jumps: bool = False
    verify_normalized: bool = False
    normalized_tol: float = 1e-5
    solver_options: Dict[str, Any] = Field(default_factory=dict)


class QutipMethodTable(
    MethodTableBase[QutipMethodTableOptions],
    QutipMixin,
    FunctionMixin,
    PrintMixin,
    ArithmeticMixin,
    BoolMixin,
    StackStoreMixin,
): ...


#######################################################################################


class QutipBackend(BackendBase):
    """
    Class representing the Qutip backend
    """

    def __init__(self):
        super().__init__()

        self.avail_checker = AvailableVariableAnalysis()
        self.type_checker = AnalogTypeChecker()
        self.dim_checker = DimensionChecker()

    def compile(self, program: str | AnalogCircuit):
        if isinstance(program, str):
            program = parse_analog(program)

        cfg = AnalogCFGBuilder()(program)
        cfg = CFGBlockAccumulator()(cfg)
        cfg = RelabelCFGBlocks()(cfg)

        return cfg

    def run(
        self,
        program: str | AnalogCircuit = "",
        *,
        options: QutipMethodTableOptions | None = None,
        **kwargs,
    ):
        """
        Method to simulate an experiment using the QuTip backend

        Args:
            program (str | AnalogCircuit): Run experiment from valid analog code or AnalogCircuit object.
            options (QutipMethodTableOptions): Options for the qutip method table
        Returns:
            Output of the QuTip simulation, Program object, CFG object, Interpreter object.
        """

        if not isinstance(program, str | AnalogCircuit):
            raise TypeError("Provide valid analog code or AnalogCircuit.")

        cfg = self.compile(program)

        self.avail_checker.analyze(cfg)
        self.type_checker.analyze(cfg)
        self.dim_checker.analyze(cfg)

        method_table = QutipMethodTable(options=options, **kwargs)
        interpreter = AnalogInterpreter(method_table=method_table)

        output = interpreter.run(cfg=cfg)

        return (output, program, cfg, interpreter)
