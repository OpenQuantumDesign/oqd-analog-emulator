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
    AnalogList,
    Annihilation,
    Bool,
    BoolAnd,
    BoolEq,
    BoolGreaterThan,
    BoolGreaterThanEq,
    BoolLessThan,
    BoolLessThanEq,
    BoolNot,
    BoolNotEq,
    BoolOr,
    Creation,
    Evolve,
    Extract,
    Identity,
    Initialize,
    MathAdd,
    MathDiv,
    MathMul,
    MathNum,
    MathImag,
    MathPow,
    MathSub,
    MathVar,
    MathFunc,
    Measure,
    ModeRegister,
    OperatorAdd,
    OperatorKron,
    OperatorMul,
    OperatorSub,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
    QuantumRegister,
)
from oqd_core.interface.analog.statement import Declaration


class QutipBackendInstructions(RewriteRule):
    def __init__(self, fock_cutoff: int = 4):
        super().__init__()
        self._fock_cutoff = fock_cutoff
    
    def map_Access(self, model: Access):
        return [('LOAD', model.name)]
    
    def map_Declaration(self, model: Declaration):
        if isinstance(model.value, AnalogList):
            return self.map_AnalogList(model.value, model.name)
        elif isinstance(model.value, QuantumRegister):
            return self.map_QuantumRegister(model.value, model.name)
        elif isinstance(model.value, ModeRegister):
            return self.map_ModeRegister(model.value, model.name)
        elif isinstance(model.value, Extract):
            return [('DEC_EX', model.name, model.value.access.name, model.value.index)]
        else:
            instructions = [('GLOBAL', model.name)] + self(model.value) + [('STORE', model.name)]
            return instructions
    
    def map_MathNum(self, model: MathNum):
        return [('CONST', model.value)]
    
    def map_MathImag(self, model: MathImag):
        return [('IMAG',)]
    
    def map_MathAdd(self, model: MathAdd):
        instructions = self(model.expr1) + self(model.expr2) + [('ADD', )]
        return instructions
    
    def map_MathSub(self, model: MathSub):
        instructions = self(model.expr1) + self(model.expr2) + [('SUB', )]
        return instructions

    def map_MathMul(self, model: MathMul):
        instructions = self(model.expr1) + self(model.expr2) + [('MUL', )]
        return instructions
    
    def map_OperatorMul(self, model: OperatorMul):
        instructions = self(model.op1) + self(model.op2) + [('MUL', )]
        return instructions
    
    def map_OperatorAdd(self, model: OperatorAdd):
        instructions = self(model.expr1) + self(model.expr2) + [('ADD', )]
        return instructions
    
    def map_OperatorSub(self, model: OperatorSub):
        instructions = self(model.expr1) + self(model.expr2) + [('SUB', )]
        return instructions

    def map_MathDiv(self, model: MathDiv):
        instructions = self(model.expr1) + self(model.expr2) + [('DIV', )]
        return instructions
    
    def map_MathPow(self, model: MathPow):
        instructions = self(model.expr1) + self(model.expr2) + [('POW', )]
        return instructions
    
    def map_MathVar(self, model: MathVar):
        return [('GLOBAL', model.name)]
    
    def map_MathFunc(self, model: MathFunc):
        if isinstance(model.expr, list):
            instructions = []
            for expr in model.expr:
                instructions += self(expr)
        else:
            instructions = self(model.expr)
        instructions += [('FUNC', model.func)]
        return instructions
    
    def map_PauliI(self, model: PauliI):
        return [('CONST', qt.qeye(2))] 
    
    def map_PauliX(self, model: PauliX):
        return [('CONST', qt.sigmax())]

    def map_PauliY(self, model: PauliY):
        return [('CONST', qt.sigmay())]

    def map_PauliZ(self, model: PauliZ):
        return [('CONST', qt.sigmaz())]

    def map_Identity(self, model: Identity):
        return [('CONST', qt.qeye(self._fock_cutoff))]

    def map_Creation(self, model: Annihilation):
        return [('CONST', qt.create(self._fock_cutoff))]

    def map_Annihilation(self, model: Creation):
        return [('CONST', qt.destroy(self._fock_cutoff))]

    def map_OperatorKron(self, model: OperatorKron):
        instructions = self(model.op1) + self(model.op2) + [('KRON', )]
        return instructions
    
    def map_Bool(self, model: Bool):
        return [('CONST', model.value)]
    
    def map_BoolNot(self, model: BoolNot):
        instructions =  self(model.expr) + [('NOT', )]
        return instructions
    
    def map_BoolAnd(self, model: BoolAnd):
        instructions = self(model.expr1) + self(model.expr2) + [('AND', )]
        return instructions
    
    def map_BoolOr(self, model: BoolOr):
        instructions = self(model.expr1) + self(model.expr2) + [('OR', )]
        return instructions
        
    def map_BoolEq(self, model: BoolEq):
        instructions = self(model.expr1) + self(model.expr2) + [('EQ', )]
        return instructions
    
    def map_BoolNotEq(self, model: BoolNotEq):
        instructions = self(model.expr1) + self(model.expr2) + [('NEQ', )]
        return instructions
        
    def map_BoolLessThan(self, model: BoolLessThan):
        instructions = self(model.expr1) + self(model.expr2) + [('LT', )]
        return instructions
        
    def map_BoolLessThanEq(self, model: BoolLessThanEq):
        instructions = self(model.expr1) + self(model.expr2) + [('LTEQ', )]
        return instructions
        
    def map_BoolGreaterThan(self, model: BoolGreaterThan):
        instructions = self(model.expr1) + self(model.expr2) + [('GT', )]
        return instructions
    
    def map_BoolGreaterThanEq(self, model: BoolGreaterThanEq):
        instructions = self(model.expr1) + self(model.expr2) + [('GTEQ', )]
        return instructions
        
    def map_QuantumRegister(self, model: QuantumRegister, name: str):
        return [('QREG', name, model.size)]
        
    def map_ModeRegister(self, model: ModeRegister, name: str):
        return [('MREG', name, model.size)]
    
    def map_AnalogList(self, model: AnalogList, name: str):
        instructions = [('GLOBAL', name)]
        for value in model.values:
            instructions += self(value)
        instructions += [('LIST', name)]
        return instructions
    
    def map_Extract(self, model: Extract):
        return [('EXTRACT', model.access.name, model.index)]
    
    def map_Evolve(self, model: Evolve):
        instructions = self(model.hamiltonian) + self(model.duration) + self(model.targets) + [('EVOLVE', )]
        return instructions
        
    def map_Initialize(self, model: Initialize):
        instructions = self(model.targets) + [('INIT', )]
        return instructions
    
    def map_Measure(self, model: Measure):
        instructions = self(model.targets) + [('MEASURE', )]
        return instructions
    
    

    
