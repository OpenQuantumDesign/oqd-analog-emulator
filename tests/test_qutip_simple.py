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

import pytest
from oqd_core.interface.analog.expr import (
    Annihilation,
    Creation,
    Identity,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
)

from oqd_analog_emulator.qutip_backend import QutipBackend


X, Y, Z, I, A, C, ID = (
    PauliX(),
    PauliY(),
    PauliZ(),
    PauliI(),
    Annihilation(),
    Creation(),
    Identity(),
)

def interpreter(program):
    backend = QutipBackend()
    program, interpreter, output = backend.run(program)
    return output

class TestQutipBackend:
    @pytest.mark.parametrize(
        "program",
        [   "r = qreg(2)",
            "list = [1, 2, 3]",
            "x = 1\n if (x > 0) {\n y = 2\n}",
            "x = 1\n if (x > 0) {\n y = 2\n} \n else {\n y = 3\n}",
        ],
    )
    def test_qutip_backend(self, program):
        output = interpreter(program)
    

# class TestQutipExperiment:
        
#     def test_one_qubit_rabi_flopping(backend):
#         """One qubit rabi flopping"""
#         source = "r = qreg(1) \n H = -(3.14 / 4) %* %X \n  evolve(H, 1, r)"
#         program, interpreter, output = backend.run(source=source)
