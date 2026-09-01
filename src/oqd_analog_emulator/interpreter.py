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

from typing import Any

from oqd_compiler_infrastructure import VisitableBaseModel
from oqd_core.analysis.utils import ControlFlowGraph
from oqd_core.interface.analog import Break, Continue

from oqd_analog_emulator.instructions import (
    AnalogInstructions,
    AnalogInstructionsCodegen,
    ListTerminators,
)

########################################################################################


AnalogVMNULL = [ListTerminators.LISTSTART, ListTerminators.LISTEND]

########################################################################################


class AnalogStack(VisitableBaseModel):
    def __init__(self):
        self.__index = []

    def __len__(self):
        return len(self.__index)

    def __str__(self):
        return str(self.__index)

    def peek(self):
        if len(self) == 0:
            return None
        return self.__index[-1]

    def push(self, item):
        if isinstance(item, list):
            self.__index.extend(reversed(item))
        else:
            self.__index.append(item)

    def pop(self):
        if len(self) == 0:
            return None

        if self.peek() is not ListTerminators.LISTSTART:
            return self.__index.pop()

        out = [self.__index.pop()]
        while True:
            curr = self.pop()
            out.append(curr)
            if curr is ListTerminators.LISTEND:
                break
        return out


class AnalogVirtualMachine:
    def __init__(self, method_table):
        self.stack = AnalogStack()
        self.store = {}
        self.registers = {}

        self.history = {}
        self.method_table = method_table

    def get_state(self, return_values):
        return self.method_table.get_state(
            return_values, self.stack, self.store, self.registers
        )

    def run(self, instructions: AnalogInstructions):
        for instruction in instructions.instructions:
            opcode = instruction.opcode.name
            args = instruction.args
            self.method_table.run(
                opcode=opcode,
                args=args,
                stack=self.stack,
                store=self.store,
                registers=self.registers,
            )

    def clear(self):
        self.stack = AnalogStack()
        self.store = {}
        self.registers = {}
        self.history = {}


class AnalogInterpreter:
    def __init__(
        self,
        method_table: Any,
        fock_cutoff: int,
    ):
        self.vm = AnalogVirtualMachine(method_table=method_table)
        self.codegen = AnalogInstructionsCodegen(fock_cutoff=fock_cutoff)

    def evaluate(self, stmt):
        instructions = self.codegen(stmt)
        self.vm.run(instructions)

    def run(self, cfg: ControlFlowGraph):
        current_block = cfg.blocks[0]

        while True:
            inverse_edge_labels = {v: k for k, v in current_block.edge_labels.items()}

            match current_block.kind:
                case "stop":
                    break
                case "start":
                    current_block = current_block.succs[0]
                case "stmt" if isinstance(current_block.stmt, (Break, Continue)):
                    current_block = cfg.blocks[
                        inverse_edge_labels.get(
                            current_block.stmt.__class__.__name__.lower()
                        )
                    ]
                case "stmt":
                    self.evaluate(current_block.stmt)
                    current_block = current_block.succs[0]
                case "branch":
                    self.evaluate(current_block.stmt)
                    cond = self.vm.stack.pop()

                    current_block = cfg.blocks[
                        inverse_edge_labels.get(str(cond).lower())
                    ]
                case _:
                    raise ValueError("Unknown kind of block in CFG")

        stack_top = self.vm.stack.pop()
        if stack_top is None:
            return []
        return self.get_state(stack_top)

    def get_store(self):
        return self.vm.store

    def get_state(self, return_values):
        return self.vm.get_state(return_values)

    def clear(self):
        self.vm.clear()
