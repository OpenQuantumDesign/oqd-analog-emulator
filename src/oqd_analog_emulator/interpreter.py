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

from oqd_analog_emulator.instructions import (
    ListTerminators,
    QutipBackendInstructions,
    QutipBackendInstructionsCodegen,
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

    def run(self, instructions: QutipBackendInstructions):
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


class AnalogInterpreter:
    def __init__(
        self,
        graph: ControlFlowGraph,
        method_table: Any,
        fock_cutoff: int,
        codegen=None,
    ):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.vm = AnalogVirtualMachine(method_table=method_table)
        self.INSTRUCTIONS = []
        self.codegen = codegen
        if codegen is None:
            self.codegen = QutipBackendInstructionsCodegen(fock_cutoff=fock_cutoff)

    def get_block(self, node: int = 0):
        return self.graph.blocks[node]

    def evaluate(self, stmt):
        instructions = self.codegen(stmt)
        self.INSTRUCTIONS.append(instructions)
        self.vm.run(instructions)

    def run(self):
        node = 1
        current_block = self.get_block(node)

        while current_block.kind != "stop":
            stmt = current_block.stmt

            if current_block.kind == "branch":
                self.evaluate(stmt)
                cond = self.vm.stack.pop()
                if cond:
                    node = next(
                        key
                        for key, val in current_block.edge_labels.items()
                        if val == "true"
                    )
                else:
                    node = next(
                        key
                        for key, val in current_block.edge_labels.items()
                        if val == "false"
                    )

            if current_block.kind == "stmt":
                if not current_block.edge_labels and stmt:
                    self.evaluate(stmt)
                if current_block.succs:
                    current_block = current_block.succs[0]
                    continue
            current_block = self.get_block(node)

        stack_top = self.vm.stack.pop()
        if stack_top is None:
            return []
        return self.get_state(stack_top)

    def get_store(self):
        return self.vm.store

    def get_state(self, return_values):
        return self.vm.get_state(return_values)

    def get_instructions(self):
        return self.INSTRUCTIONS
