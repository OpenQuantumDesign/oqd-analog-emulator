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

import qutip as qt
from oqd_compiler_infrastructure import RewriteRule
from oqd_core.interface.analog.expr import (
    Access,
    Annihilation,
    Creation,
    Identity,
    MathAdd,
    MathDiv,
    MathMul,
    MathNum,
    MathSub,
    OperatorKron,
    OperatorMul,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
)
from oqd_core.interface.analog.statement import Declaration


class QutipBackendInstructions(RewriteRule):
    def __init__(self):
        super().__init__()
        self._fock_cutoff = 4
    
    def map_Access(self, model: Access):
        return [('LOAD', model.name)]
    
    def map_Declaration(self, model: Declaration):
        instructions = [('GLOBALI', model.name)] + self(model.value) + [('STORE', model.name)]
        return instructions
    
    def map_MathNum(self, model: MathNum):
        return [('CONSTI', model.value)]
    
    def map_MathAdd(self, model: MathAdd):
        instructions = self(model.expr1) + self(model.expr2) + [('ADDI', )]
        return instructions
    
    def map_MathSub(self, model: MathSub):
        instructions = self(model.expr1) + self(model.expr2) + [('SUBI', )]
        return instructions

    def map_MathMul(self, model: MathMul):
        instructions = self(model.expr1) + self(model.expr2) + [('MULI', )]
        return instructions
    
    def map_OperatorMul(self, model: OperatorMul):
        instructions = self(model.op1) + self(model.op2) + [('MULI', )]
        return instructions

    def map_MathDiv(self, model: MathDiv):
        instructions = self(model.expr1) + self(model.expr2) + [('DIVI', )]
        return instructions
    
    def map_PauliI(self, model: PauliI):
        return [('CONSTI', qt.qeye(2))] 
    
    def map_PauliX(self, model: PauliX):
        return [('CONSTI', qt.sigmax())]

    def map_PauliY(self, model: PauliY):
        return [('CONSTI', qt.sigmay())]

    def map_PauliZ(self, model: PauliZ):
        return [('CONSTI', qt.sigmaz())]

    def map_Identity(self, model: Identity):
        return [('CONSTI', qt.qeye(self._fock_cutoff))]

    def map_Creation(self, model: Annihilation):
        return [('CONSTI', qt.create(self._fock_cutoff))]

    def map_Annihilation(self, model: Creation):
        return [('CONSTI', qt.destroy(self._fock_cutoff))]

    def map_OperatorKron(self, model: OperatorKron):
        instructions = self(model.op1) + self(model.op2) + [('KRONI', )]
        return instructions
    
