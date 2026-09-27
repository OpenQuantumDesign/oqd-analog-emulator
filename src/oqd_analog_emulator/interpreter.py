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


from collections.abc import MutableMapping, MutableSequence
from typing import Any, List

import numpy as np
from oqd_compiler_infrastructure import CFG

from oqd_analog_emulator.instructions import (
    AnalogInstructions,
    AnalogInstructionsCodegen,
    ListTerminators,
)
from oqd_analog_emulator.method_table import (
    MethodTableBase,
    MethodTableOptionsBase,
    MethodTableRegistry,
    QuantumRegister,
    RegisterName,
)

########################################################################################


class AnalogStack(MutableSequence[Any]):
    def __init__(self):
        self._stack = []

    def __repr__(self):
        return self._stack.__repr__()

    def __str__(self):
        return self._stack.__str__()

    def __len__(self):
        return len(self._stack)

    def __getitem__(self, idx):
        return self._stack[idx]

    def __setitem__(self, idx, value):
        self._stack[idx] = value

    def __delitem__(self, idx):
        del self._stack[idx]

    def insert(self, idx, value):
        self._stack.insert(idx, value)

    def peek(self):
        if len(self) == 0:
            return None
        return self[-1]

    def push(self, item):
        if isinstance(item, list):
            self.extend(reversed(item))
        else:
            self.append(item)

    def pop(self):
        if len(self) == 0:
            return None

        out = self._stack.pop()

        if out is not ListTerminators.LISTSTART:
            return out

        out = [out]
        while True:
            curr = self.pop()
            out.append(curr)
            if curr is ListTerminators.LISTEND:
                break

        return out


class AnalogRegisters(MutableMapping[RegisterName, QuantumRegister]):
    def __init__(self):
        self._registers = {}
        self._current = 0

    def __repr__(self):
        return self._registers.__repr__()

    def __str__(self):
        return self._registers.__str__()

    def __len__(self):
        return len(self._registers)

    def __getitem__(self, key: RegisterName):
        if not isinstance(key, RegisterName):
            raise KeyError(
                f"Keys of AnalogRegisters should be of type RegisterName, got {type(key).__qualname__}"
            )

        return self._registers[key]

    def __setitem__(self, key: RegisterName, value: QuantumRegister):
        if not isinstance(key, RegisterName):
            raise KeyError(
                f"Keys of AnalogRegisters should be of type RegisterName, got {type(key).__qualname__}"
            )

        if not isinstance(value, QuantumRegister):
            raise ValueError(
                f"Values in AnalogRegisters should be of type QuantumRegister, got {type(value).__qualname__}"
            )

        self._registers[key] = value

    def __delitem__(self, key: RegisterName):
        if not isinstance(key, RegisterName):
            raise KeyError(
                f"Keys of AnalogRegisters should be of type RegisterName, got {type(key).__qualname__}"
            )

        del self._registers[key]

    def __iter__(self):
        return self._registers.__iter__()

    def create(self, size, dim, t):
        names = []
        for _ in range(size):
            name = RegisterName(name="r", index=self._current, dim=dim)
            self[name] = QuantumRegister(
                name=[name], time=t, time_last_updated=t, state=None
            )
            names.append(name)
            self._current += 1

        return names

    def wipe(self, names: List[RegisterName]):
        for name in names:
            del self[name]

    @property
    def names(self):
        return set(map(lambda k: k.name, self.keys()))

    def contains_name(self, name):
        return name in self.names


########################################################################################


class AnalogVirtualMachine:
    def __init__(
        self,
        *,
        method_table: str | MethodTableBase,
        options: MethodTableOptionsBase | None = None,
        **kwargs,
    ):
        self.stack = AnalogStack()
        self.store = {
            "$abs": np.abs,
            "$real": np.real,
            "$imag": np.imag,
            "$conj": np.conj,
            "$sin": np.sin,
            "$cos": np.cos,
            "$tan": np.tan,
            "$exp": np.exp,
            "$log": np.log,
            "$sinh": np.sinh,
            "$cosh": np.cosh,
            "$tanh": np.tanh,
            "$asin": np.asin,
            "$acos": np.acos,
            "$atan": np.atan,
            "$atan2": np.atan2,
            "$asinh": np.asinh,
            "$acosh": np.acosh,
            "$atanh": np.atanh,
            "$heaviside": lambda x: np.heaviside(x, 1),
            "$round": lambda x: np.round(x).astype(int),
            "#t": lambda t, s: t,
            "#s": lambda t, s: s,
        }
        self.registers = AnalogRegisters()
        self.machine_time = 0.0

        self.history = {}

        match method_table:
            case str():
                self.method_table = MethodTableRegistry[method_table](
                    options=options, **kwargs
                )
            case MethodTableBase():
                self.method_table = method_table
            case _:
                raise ValueError(f"Invalid method table ({method_table})")

    def get_state(self, return_values):
        return self.method_table.get_state(return_values, self)

    def run(self, instructions: AnalogInstructions):
        for instruction in instructions.instructions:
            opcode = instruction.opcode.name
            args = instruction.args
            self.method_table.run(opcode=opcode, args=args, vm=self)

    def reset(self):
        self.stack = AnalogStack()
        self.store = {}
        self.registers = AnalogRegisters()
        self.history = {}


class AnalogInterpreter:
    def __init__(
        self,
        *,
        method_table: str | MethodTableBase,
        options: MethodTableOptionsBase | None = None,
        **kwargs,
    ):
        self.vm = AnalogVirtualMachine(
            method_table=method_table, options=options, **kwargs
        )
        self.codegen = AnalogInstructionsCodegen()

    def evaluate(self, stmt):
        instructions = self.codegen(stmt)
        self.vm.run(instructions)

    def run(self, cfg: CFG):
        current_block = cfg.blocks[0]

        while current_block:
            for stmt in current_block.stmts:
                self.evaluate(stmt)

            if current_block.succs == set():
                break

            if current_block.edge_labels:
                cond = str(self.vm.stack.pop()).lower()
                current_block = cfg[current_block.edge_labels[cond]]
                continue

            current_block = cfg[next(iter(current_block.succs))]

        stack_top = self.vm.stack.pop()
        if stack_top is None:
            return []
        return self.get_state(stack_top)

    def get_store(self):
        return self.vm.store

    def get_state(self, return_values):
        return self.vm.get_state(return_values)

    def reset(self):
        self.vm.reset()
