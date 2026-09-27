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

from __future__ import annotations

from collections.abc import MutableSequence
from enum import Enum, auto
from typing import Any

from oqd_compiler_infrastructure import RewriteRule, TypeReflectBaseModel
from oqd_core.interface.analog import (
    Access,
    Add,
    AnalogList,
    And,
    Annihilation,
    BuiltinCall,
    Complex,
    Constant,
    Creation,
    Declaration,
    Div,
    Eq,
    Evolve,
    Extract,
    Geq,
    Gt,
    Identity,
    Initialize,
    Kron,
    Leq,
    Lt,
    Measure,
    ModeRegister,
    Mul,
    Neg,
    Neq,
    Not,
    Or,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
    Pos,
    Pow,
    QuantumRegister,
    RuntimeVar,
    Sub,
    Xor,
)
from oqd_core.interface.analog.expr import BinaryOp, Ladder, Operator, Pauli, UnaryOp
from pydantic import (
    ConfigDict,
    model_validator,
)

########################################################################################


class ListTerminators(Enum):
    LISTSTART = auto()
    LISTEND = auto()


class OpCode(Enum):
    # Stack
    CONST = auto()  # Adds item to stack

    # Store
    GLOBAL = auto()  # Allocates space on store for variable
    STORE = auto()  # Store value to store
    LOAD = auto()  # Load value from store
    LOADV = auto()  # Load runtime value

    # List
    EXTRACT = auto()  # Extract element from list in store

    # Arith
    NEG = auto()  # Negative
    POS = auto()  # Positive
    ADD = auto()  # Addition
    MUL = auto()  # Multiply
    SUB = auto()  # Subtract
    DIV = auto()  # Divide
    POW = auto()  # Exponentiation
    KRON = auto()  # Kronecker product

    # Bool
    NOT = auto()  # Logical Not
    AND = auto()  # Logical And
    OR = auto()  # Logical Or
    XOR = auto()  # Logical Xor

    # Compare
    EQ = auto()  # Equal
    NEQ = auto()  # Not Equal
    LT = auto()  # Less than
    LEQ = auto()  # Less than equal
    GT = auto()  # Greater than
    GEQ = auto()  # Greater than equal

    # Pauli
    PI = auto()  # Pauli Identity
    PX = auto()  # Pauli X
    PY = auto()  # Pauli Y
    PZ = auto()  # Pauli Z

    # Mode
    MA = auto()  # Annihilation
    MC = auto()  # Creation
    MI = auto()  # Mode Identity

    # Quantum reg
    QREG = auto()  # Discrete qudit quantum register
    MREG = auto()  # Infinite mode quantum register

    # Quantum op
    INIT = auto()  # Initialize quantum targets
    EVOLVE = auto()  # Evolve quantum targets
    MEASURE = auto()  # Measure quantum targets

    # Func
    MFUNC = auto()  # Math functions
    LEN = auto()  # length of array
    RANGE = auto()  # create a list with a range of values
    FLATTEN = auto()  # Flatten a list of list

    # @property
    # def num_args(self):
    #     match self:
    #         case _ if self is OpCode.QREG:
    #             return 3
    #         case _ if self in [OpCode.EXTRACT, OpCode.MREG]:
    #             return 2
    #         case _ if self in [
    #             OpCode.LOAD,
    #             OpCode.GLOBAL,
    #             OpCode.FUNC,
    #             OpCode.CONST,
    #             OpCode.STORE,
    #             OpCode.EXTRACT,
    #             OpCode.MREG,
    #             OpCode.QREG,
    #         ]:
    #             return 1
    #         case _:
    #             return 0

    @staticmethod
    def from_ast(op):
        match op:
            case PauliI():
                return OpCode.PI
            case PauliX():
                return OpCode.PX
            case PauliY():
                return OpCode.PY
            case PauliZ():
                return OpCode.PZ
            case Annihilation():
                return OpCode.MA
            case Creation():
                return OpCode.MC
            case Identity():
                return OpCode.MI
            case Pos():
                return OpCode.POS
            case Neg():
                return OpCode.NEG
            case Add():
                return OpCode.ADD
            case Sub():
                return OpCode.SUB
            case Mul():
                return OpCode.MUL
            case Div():
                return OpCode.DIV
            case Pow():
                return OpCode.POW
            case Kron():
                return OpCode.KRON
            case Not():
                return OpCode.NOT
            case And():
                return OpCode.AND
            case Or():
                return OpCode.OR
            case Xor():
                return OpCode.XOR
            case Eq():
                return OpCode.EQ
            case Neq():
                return OpCode.NEQ
            case Lt():
                return OpCode.LT
            case Leq():
                return OpCode.LEQ
            case Gt():
                return OpCode.GT
            case Geq():
                return OpCode.GEQ
            case BuiltinCall(func="len"):
                return OpCode.LEN
            case BuiltinCall(func="range"):
                return OpCode.RANGE
            case BuiltinCall(func="flatten"):
                return OpCode.FLATTEN
        raise ValueError()


AnalogVMNULL = [ListTerminators.LISTSTART, ListTerminators.LISTEND]


########################################################################################


class AnalogInstructions(MutableSequence, TypeReflectBaseModel):
    instructions: list[AnalogInstruction] = []

    def __getitem__(self, idx):
        return self.instructions[idx]

    def __setitem__(self, idx, value):
        self.instructions[idx] = value

    def __delitem__(self, idx):
        del self.instructions[idx]

    def __len__(self):
        return len(self.instructions)

    def insert(self, idx, value):
        self.instructions.insert(idx, value)

    def __add__(self, other):
        if isinstance(other, AnalogInstruction):
            return AnalogInstructions(instructions=self.instructions + [other])
        return AnalogInstructions(instructions=self.instructions + other.instructions)

    def __radd__(self, other):
        if isinstance(other, AnalogInstruction):
            return AnalogInstructions(instructions=[other] + self)
        return AnalogInstructions(instructions=other.instructions + self.instructions)


class AnalogInstruction(TypeReflectBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    opcode: OpCode
    args: list[Any] = []

    def __add__(self, other):
        if isinstance(other, AnalogInstruction):
            return AnalogInstructions(instructions=[self, other])


########################################################################################


class AnalogInstructionsCodegen(RewriteRule):
    @staticmethod
    def const(value, *, single=False):
        instr = AnalogInstruction(opcode=OpCode.CONST, args=[value])
        return instr if single else AnalogInstructions(instructions=[instr])

    @staticmethod
    def op(code):
        return AnalogInstructions(instructions=[AnalogInstruction(opcode=code)])

    ########################################################################################

    # Variables

    def map_Access(self, model: Access):
        out = self.const(model.name)
        out += self.op(OpCode.LOAD)
        return out

    def map_Declaration(self, model: Declaration):
        out = self.const(model.name)
        out += self.op(OpCode.GLOBAL)
        out += (
            self.const(f"&{model.value.name}")
            if isinstance(model.value, Access)
            else self(model.value)
        )
        out += self.const(model.name)
        out += self.op(OpCode.STORE)
        return out

    def map_RuntimeVar(self, model: RuntimeVar):
        out = self.const(f"{model.name}")
        out += self.op(OpCode.LOADV)
        return out

    ########################################################################################

    # Constant

    def map_Constant(self, model: Constant):
        value = model.value
        if isinstance(value, Complex):
            value = value.real + 1j * value.imag

        return self.const(value)

    def map_Pauli(self, model: Pauli):
        out = self(model.dim)
        out += self(model.level2)
        out += self(model.level1)
        out += self.op(OpCode.from_ast(model))
        return out

    def map_Ladder(self, model: Ladder):
        return self.op(OpCode.from_ast(model))

    ########################################################################################

    # Arithmetic & Boolean Operation

    def map_BinaryOp(self, model: BinaryOp):
        out = self(model.exprs[-1])
        for e in reversed(model.exprs[:-1]):
            out += self(e)
            out += self.op(OpCode.from_ast(model))
        return out

    def map_UnaryOp(self, model: UnaryOp):
        out = self(model.expr)
        out += self.op(OpCode.from_ast(model))
        return out

    ########################################################################################

    # List

    def map_AnalogList(self, model: AnalogList):
        out = self.const(ListTerminators.LISTEND)
        for value in reversed(model.values):
            out += self(value)
        out += self.const(ListTerminators.LISTSTART)
        return out

    def map_Extract(self, model: Extract):
        out = self(model.index)
        out += self.const(model.access.name)
        out += self.op(OpCode.EXTRACT)
        return out

    ########################################################################################

    # Functions

    def map_BuiltinCall(self, model: BuiltinCall):
        out = self(model.args[-1])

        for arg in reversed(model.args[:-1]):
            out += self(arg)

        match model.func:
            case "len" | "flatten":
                out += self.op(OpCode.from_ast(model))
            case "range" if len(model.args) == 1:
                out.insert(-2, self.const(1, single=True))
                out += self.const(0)
                out += self.op(OpCode.from_ast(model))
            case "range" if len(model.args) == 2:
                out += self.const(1)
                out += self.op(OpCode.from_ast(model))
            case "range" if len(model.args) == 3:
                out += self.op(OpCode.from_ast(model))
            case (
                "abs"
                | "real"
                | "imag"
                | "conj"
                | "sin"
                | "cos"
                | "tan"
                | "atan2"
                | "exp"
                | "log"
                | "sinh"
                | "cosh"
                | "tanh"
                | "atan"
                | "acos"
                | "asin"
                | "atanh"
                | "asinh"
                | "acosh"
                | "heaviside"
                | "round"
            ):
                out += self.const(f"${model.func}")
                out += (
                    self.op(OpCode.FUNC)
                    if model.func
                    in [
                        "range",
                        "len",
                        "flatten",
                    ]
                    else self.op(OpCode.MFUNC)
                )
            case _:
                raise ValueError()
        return out

    ########################################################################################

    # Quantum Register

    def map_QuantumRegister(self, model: QuantumRegister):
        out = self(model.dim)
        out += self(model.size)
        out += self.op(OpCode.QREG)
        return out

    def map_ModeRegister(self, model: ModeRegister):
        out = self(model.size)
        out += self.op(OpCode.MREG)
        return out

    ########################################################################################

    # Quantum Operation

    def map_Evolve(self, model: Evolve):
        out = self(model.targets)
        out += self(model.duration)
        out += self(model.jumps)
        out += self(model.hamiltonian)
        out += self.op(OpCode.EVOLVE)
        return out

    def map_Initialize(self, model: Initialize):
        out = self(model.targets)
        out += self.op(OpCode.INIT)
        return out

    def map_Measure(self, model: Measure):
        out = self(model.targets)
        out += self.op(OpCode.MEASURE)
        return out
