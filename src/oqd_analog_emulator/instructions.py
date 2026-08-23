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

from enum import Enum
from typing import Any

import qutip as qt
from oqd_compiler_infrastructure import RewriteRule, TypeReflectBaseModel
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
    Identifier,
    Identity,
    Initialize,
    MathAdd,
    MathDiv,
    MathFunc,
    MathImag,
    MathMul,
    MathNum,
    MathPow,
    MathSub,
    MathVar,
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
from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)


class ListTerminators(Enum):
    LISTSTART = 0
    LISTEND = 1

class ALIAS(BaseModel):
    target: Identifier
    
class OpCode(Enum):
    GLOBAL = 0
    LOAD = 1
    EXTRACT = 2
    CONST = 3
    STORE = 4
    ADD = 5
    MUL = 6
    SUB = 7
    DIV = 8
    POW = 9
    FUNC = 10
    KRON = 11
    IMAG = 12
    NOT = 13
    AND = 14
    OR = 15
    EQ = 16
    NEQ = 17
    LT = 18
    LTEQ = 19
    GT = 20
    GTEQ = 21
    EVOLVE = 22
    INIT = 23
    MEASURE = 24
    QREG = 25
    MREG = 26
    
    @property
    def num_args(self):
        match self:
            case _ if self in [OpCode.EXTRACT, OpCode.QREG, OpCode.MREG]:
                return 2
            case _ if self in [OpCode.LOAD, OpCode.GLOBAL, OpCode.FUNC, OpCode.CONST, OpCode.STORE]:
                return 1
            case _:
                return 0


class QutipBackendInstruction(TypeReflectBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    opcode: OpCode
    args: list[Any] = []
    
    @model_validator(mode='after')
    def validate_args_num(self):
        # print(value)
        if self.opcode.num_args != len(self.args):
            raise ValueError(...)
        return self
    
class QutipBackendInstructions(TypeReflectBaseModel):
    instructions: list[QutipBackendInstruction] = []

    def __add__(self, other):
        if isinstance(other, QutipBackendInstruction):
            return QutipBackendInstructions(instructions = self.instructions + [other])
        return QutipBackendInstructions(instructions = self.instructions + other.instructions)
    
    def __radd__(self, other):
        if isinstance(other, QutipBackendInstruction):
            return QutipBackendInstructions(instructions = [other] + self)
        return QutipBackendInstructions(instructions = other.instructions + self.instructions)
        
    
def _is_constant_math(model) -> bool:
    if isinstance(model, (Access, MathNum, MathImag)):
        return True
    if isinstance(model, MathVar):
        return False
    if isinstance(model, MathFunc):
        arg = model.expr
        if isinstance(arg, list):
            return all(_is_constant_math(a) for a in arg)
        return _is_constant_math(arg)
    if isinstance(model, (MathAdd, MathSub, MathMul, MathDiv, MathPow)):
        return _is_constant_math(model.expr1) and _is_constant_math(model.expr2)
    if isinstance(model, (OperatorAdd, OperatorKron, OperatorMul, OperatorSub)):
        return _is_constant_math(model.op1) and _is_constant_math(model.op2)
    return True


class QutipBackendInstructionsCodegen(RewriteRule):
    def __init__(self, fock_cutoff: int = 4):
        super().__init__()
        self._fock_cutoff = fock_cutoff
    
    def map_Access(self, model: Access):
        instruction = QutipBackendInstruction(opcode=OpCode.LOAD, args=[model.name])
        return QutipBackendInstructions(instructions=[instruction])
    
    def map_Declaration(self, model: Declaration):
        if not _is_constant_math(model):
            instruction = QutipBackendInstruction(opcode=OpCode.CONST, args=[model])
            return QutipBackendInstructions(instructions=[instruction])
        # if isinstance(model.value, AnalogList):
        #     instruction = QutipBackendInstruction(opcode=OpCode.GLOBAL, args=[model.name])
        #     return self.map_AnalogList(model.value, model.name)
        if isinstance(model.value, QuantumRegister):
            return self.map_QuantumRegister(model.value, model.name)
        if isinstance(model.value, ModeRegister):
            return self.map_ModeRegister(model.value, model.name)
        
        instr1 = QutipBackendInstruction(opcode=OpCode.GLOBAL, args=[model.name])
        if isinstance(model.value, Access):
            instr2 = QutipBackendInstruction(opcode=OpCode.CONST, args=[ALIAS(target=model.name)])
        else:
            instr2 = self(model.value)
        instr3 = QutipBackendInstruction(opcode=OpCode.STORE, args=[model.name])
        instructions = QutipBackendInstructions(instructions=[instr1]) + instr2 + instr3
        return instructions
        
    
    def map_MathNum(self, model: MathNum):
        if not _is_constant_math(model):
            instruction = QutipBackendInstruction(opcode=OpCode.CONST, args=[model])
        else:
            instruction = QutipBackendInstruction(opcode=OpCode.CONST, args=[model.value])
        return QutipBackendInstructions(instructions=[instruction])
    
    def map_MathImag(self, model: MathImag):
        if not _is_constant_math(model):
            instruction = QutipBackendInstruction(opcode=OpCode.CONST, args=[model])
        else:
            instruction = QutipBackendInstruction(opcode=OpCode.CONST, args=[1j])
        return QutipBackendInstructions(instructions=[instruction])
    
    def map_MathAdd(self, model: MathAdd):
        instructions = QutipBackendInstructions()
        if not _is_constant_math(model):
            instructions += QutipBackendInstruction(opcode=OpCode.CONST, args=[model])
        else:
            instructions += self(model.expr1) + self(model.expr2)
            instructions += QutipBackendInstruction(opcode=OpCode.ADD)
        return instructions
    
    def map_MathSub(self, model: MathSub):
        instructions = QutipBackendInstructions()
        if not _is_constant_math(model):
            instructions += QutipBackendInstruction(opcode=OpCode.CONST, args=[model])
        else: 
            instructions += self(model.expr1) + self(model.expr2)
            instructions += QutipBackendInstruction(opcode=OpCode.SUB)
        return instructions

    def map_MathMul(self, model: MathMul):
        instructions = QutipBackendInstructions()
        if not _is_constant_math(model):
            instructions += QutipBackendInstruction(opcode=OpCode.CONST, args=[model])
        else:
            instructions += self(model.expr1) + self(model.expr2)
            instructions += QutipBackendInstruction(opcode=OpCode.MUL)
        return instructions
    
    def map_OperatorMul(self, model: OperatorMul):
        instructions = QutipBackendInstructions()
        if not _is_constant_math(model):
            instructions += QutipBackendInstruction(opcode=OpCode.CONST, args=[model])
        else:
            instructions += self(model.op1) + self(model.op2)
            instructions += QutipBackendInstruction(opcode=OpCode.MUL)
        return instructions
    
    def map_OperatorAdd(self, model: OperatorAdd):
        instructions = QutipBackendInstructions()
        if not _is_constant_math(model):
            instructions += QutipBackendInstruction(opcode=OpCode.CONST, args=[model])
        else:
            instructions += self(model.expr1) + self(model.expr2)
            instructions += QutipBackendInstruction(opcode=OpCode.ADD)
        return instructions
    
    def map_OperatorSub(self, model: OperatorSub):
        instructions = QutipBackendInstructions()
        if not _is_constant_math(model):
            instructions += QutipBackendInstruction(opcode=OpCode.CONST, args=[model])
        else:
            instructions += self(model.expr1) + self(model.expr2)
            instructions += QutipBackendInstruction(opcode=OpCode.SUB)
        return instructions

    def map_MathDiv(self, model: MathDiv):
        instructions = QutipBackendInstructions()
        if not _is_constant_math(model):
            instructions += QutipBackendInstruction(opcode=OpCode.CONST, args=[model])
        else:
            instructions += self(model.expr1) + self(model.expr2)
            instructions += QutipBackendInstruction(opcode=OpCode.DIV)
        return instructions
    
    def map_MathPow(self, model: MathPow):
        instructions = QutipBackendInstructions()
        if not _is_constant_math(model):
            instructions += QutipBackendInstruction(opcode=OpCode.CONST, args=[model])
        else:
            instructions += self(model.expr1) + self(model.expr2)
            instructions += QutipBackendInstruction(opcode=OpCode.POW)
        return instructions
    
    def map_MathVar(self, model: MathVar):
        instruction = QutipBackendInstruction(opcode=OpCode.CONST, args=[model])
        return QutipBackendInstructions(instructions=[instruction])
    
    def map_MathFunc(self, model: MathFunc):
        if not _is_constant_math(model):
            instruction = QutipBackendInstruction(opcode=OpCode.CONST, args=[model])
            return QutipBackendInstructions(instructions=[instruction])
        if isinstance(model.expr, list):
            instructions = QutipBackendInstructions()
            for expr in model.expr:
                instructions += self(expr)
        else:
            instructions = self(model.expr)
        instr1 = QutipBackendInstruction(opcode=OpCode.FUNC, args=[model.func])
        instructions +=  instr1
        return instructions
    
    def map_PauliI(self, model: PauliI):
        instruction = QutipBackendInstruction(opcode=OpCode.CONST, args=[qt.qeye(2)])
        return QutipBackendInstructions(instructions=[instruction])
    
    def map_PauliX(self, model: PauliX):
        instruction = QutipBackendInstruction(opcode=OpCode.CONST, args=[qt.sigmax()])
        return QutipBackendInstructions(instructions=[instruction])

    def map_PauliY(self, model: PauliY):
        instruction = QutipBackendInstruction(opcode=OpCode.CONST, args=[qt.sigmay()])
        return QutipBackendInstructions(instructions=[instruction])
    
    def map_PauliZ(self, model: PauliZ):
        instruction = QutipBackendInstruction(opcode=OpCode.CONST, args=[qt.sigmaz()])
        return QutipBackendInstructions(instructions=[instruction])

    def map_Identity(self, model: Identity):
        instruction = QutipBackendInstruction(opcode=OpCode.CONST, args=[qt.qeye(self._fock_cutoff)])
        return QutipBackendInstructions(instructions=[instruction])

    def map_Creation(self, model: Annihilation):
        instruction = QutipBackendInstruction(opcode=OpCode.CONST, args=[qt.create(self._fock_cutoff)])
        return QutipBackendInstructions(instructions=[instruction])

    def map_Annihilation(self, model: Creation):
        instruction = QutipBackendInstruction(opcode=OpCode.CONST, args=[qt.destroy(self._fock_cutoff)])
        return QutipBackendInstructions(instructions=[instruction])

    def map_OperatorKron(self, model: OperatorKron):
        if not _is_constant_math(model):
            instruction = QutipBackendInstruction(opcode=OpCode.CONST, args=[model])
            return QutipBackendInstructions(instructions=[instruction])
        instr1 = QutipBackendInstruction(opcode=OpCode.KRON)
        instructions = self(model.op1) + self(model.op2) + instr1
        return instructions
    
    def map_Bool(self, model: Bool):
        instruction = QutipBackendInstruction(opcode=OpCode.CONST, args=[model.value])
        return QutipBackendInstructions(instructions=[instruction])
    
    def map_BoolNot(self, model: BoolNot):
        instr1 = QutipBackendInstruction(opcode=OpCode.NOT)
        instructions =  self(model.expr) + instr1
        return instructions
    
    def map_BoolAnd(self, model: BoolAnd):
        instr1 = QutipBackendInstruction(opcode=OpCode.AND)
        instructions = self(model.expr1) + self(model.expr2) + instr1
        return instructions
    
    def map_BoolOr(self, model: BoolOr):
        instr1 = QutipBackendInstruction(opcode=OpCode.OR)
        instructions = self(model.expr1) + self(model.expr2) + instr1
        return instructions
        
    def map_BoolEq(self, model: BoolEq):
        instr1 = QutipBackendInstruction(opcode=OpCode.EQ)
        instructions = self(model.expr1) + self(model.expr2) + instr1
        return instructions
    
    def map_BoolNotEq(self, model: BoolNotEq):
        instr1 = QutipBackendInstruction(opcode=OpCode.NEQ)
        instructions = self(model.expr1) + self(model.expr2) + instr1
        return instructions
        
    def map_BoolLessThan(self, model: BoolLessThan):
        instr1 = QutipBackendInstruction(opcode=OpCode.LT)
        instructions = self(model.expr1) + self(model.expr2) + instr1
        return instructions
        
    def map_BoolLessThanEq(self, model: BoolLessThanEq):
        instr1 = QutipBackendInstruction(opcode=OpCode.LTEQ)
        instructions = self(model.expr1) + self(model.expr2) + instr1
        return instructions
        
    def map_BoolGreaterThan(self, model: BoolGreaterThan):
        instr1 = QutipBackendInstruction(opcode=OpCode.GT)
        instructions = self(model.expr1) + self(model.expr2) + instr1
        return instructions
    
    def map_BoolGreaterThanEq(self, model: BoolGreaterThanEq):
        instr1 = QutipBackendInstruction(opcode=OpCode.GTEQ)
        instructions = self(model.expr1) + self(model.expr2) + instr1
        return instructions
        
    def map_QuantumRegister(self, model: QuantumRegister, name: str):
        instruction = QutipBackendInstruction(opcode=OpCode.QREG, args=[name, model.size])
        return QutipBackendInstructions(instructions=[instruction])
        
    def map_ModeRegister(self, model: ModeRegister, name: str):
        instruction = QutipBackendInstruction(opcode=OpCode.MREG, args=[name, model.size])
        return QutipBackendInstructions(instructions=[instruction])
    
    def map_AnalogList(self, model: AnalogList):
        instructions = QutipBackendInstructions()
        instructions += QutipBackendInstruction(opcode=OpCode.CONST, args=[ListTerminators.LISTEND])
        for i in list(range(len(model.values)-1, -1, -1)):
            value = model.values[i]
            instructions += self(value)
        instructions += QutipBackendInstruction(opcode=OpCode.CONST, args=[ListTerminators.LISTSTART])
        return instructions
    
    def map_Extract(self, model: Extract):
        instruction = QutipBackendInstruction(opcode=OpCode.EXTRACT, args=[model.access.name, model.index])
        return QutipBackendInstructions(instructions=[instruction])
    
    def map_Evolve(self, model: Evolve):
        instr1 = QutipBackendInstruction(opcode=OpCode.EVOLVE)
        instructions = self(model.hamiltonian) + self(model.duration) + self(model.targets) + instr1
        return instructions
        
    def map_Initialize(self, model: Initialize):
        instr1 = QutipBackendInstruction(opcode=OpCode.INIT)
        instructions = self(model.targets) + instr1
        return instructions
    
    def map_Measure(self, model: Measure):
        instr1 = QutipBackendInstruction(opcode=OpCode.MEASURE)
        instructions = self(model.targets) + instr1
        return instructions
    
    

    
