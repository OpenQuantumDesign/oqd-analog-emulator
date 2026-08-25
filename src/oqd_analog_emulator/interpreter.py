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

import itertools
import math
import time

import numpy as np
import qutip as qt
from oqd_compiler_infrastructure import Post
from oqd_core.analysis.utils import ControlFlowGraph
from oqd_core.interface.analog.expr import MathExpr, OperatorExpr
from pydantic import BaseModel

from oqd_analog_emulator.instructions import (
    ALIAS,
    ListTerminators,
    QutipBackendInstructions,
    QutipBackendInstructionsCodegen,
)
from oqd_analog_emulator.passes import QutipQobjEvoGenerator


class RegisterObject(BaseModel):
    name: str
    index: int

    def __hash__(self):
        return hash((self.name, self.index))


class QubitObject:
    register: RegisterObject
    time: int
    state: object


class QubitRegister:
    register: list[RegisterObject] = []
    time: int
    state: object

    @property
    def n(self):
        return len(self.register)


class ModeObject:
    register: RegisterObject
    time: int
    state: object


QutipVMNULL = [ListTerminators.LISTSTART, ListTerminators.LISTEND]


def recursive_filter(l, cond):
    return list(
        map(
            lambda x: recursive_filter(x) if isinstance(x, list) else x, filter(cond, l)
        )
    )


class QutipVM:
    def __init__(self, n_shots=10, fock_cutoff=4, dt=0.1):
        self._n_shots = n_shots
        self._fock_cutoff = fock_cutoff
        self._dt = dt
        self.stack = []
        self.store = {}
        self.registers = {}
        self.GLOBAL_T = 0.0
        self.history = {}

    def get_store(self):
        return self.store

    def get_state(self, return_values, *, verbose=False):
        if not isinstance(return_values, list):
            return return_values
        out = []
        for value in return_values:
            if isinstance(value, ListTerminators):
                continue
            if isinstance(value, list):
                out.append(self.get_state(value, verbose=verbose))
            elif isinstance(value, RegisterObject):
                out.append(
                    (value, self.registers[value]) if verbose else self.registers[value]
                )
            else:
                out.append(value)
        return out

    def get_args(
        self,
        num: int,
    ):
        out = []
        for _ in list(range(num)):
            item = self.pop()
            if isinstance(item, RegisterObject):
                out.append([self.registers[item]])
            elif isinstance(item, list):
                out.append(
                    recursive_filter(item, lambda x: not isinstance(x, ListTerminators))
                )
            else:
                out.append(item)
        # print(self.get_state(out))
        return self.get_state(out)

    def push(self, item):
        if isinstance(item, list):
            self.stack.extend(reversed(item))
        else:
            self.stack.append(item)

    def pop(self):
        if not self.stack:
            return None

        if self.stack[-1] is not ListTerminators.LISTSTART:
            return self.stack.pop()

        out = [self.stack.pop()]
        while True:
            curr = self.pop()
            out.append(curr)
            if curr is ListTerminators.LISTEND:
                break
        return out

    def run(self, instructions: QutipBackendInstructions):
        for instruction in instructions.instructions:
            opcode = instruction.opcode.name
            args = instruction.args
            getattr(self, f"run_{opcode}")(*args)

    def run_GLOBAL(self, name):
        if name not in self.store:
            self.store[name] = None

    def run_CONST(self, value):
        self.push(value)

    def run_STORE(self, name):
        self.store[name] = self.pop()

    def run_LOAD(self, name):
        if isinstance(name, ALIAS):
            self.run_LOAD(name.target)
        if isinstance(name, RegisterObject):
            self.push(self.registers[name])
        if name in self.store:
            item = self.store[name]
            self.push(item)
        else:
            raise ValueError

    def run_FUNC(self, func):
        output = None
        operation = getattr(math, func, None)
        if operation is None:
            operation = getattr(np, func, None)
        if operation is None:
            raise ValueError("Unknown math function")

        match func:
            case "abs":
                output = abs(self.get_args(1)[0])
            case "heaviside":
                output = np.heaviside(self.get_args(1)[0], 0)
            case "atan2":
                args = self.get_args(2)
                x = args[0]
                y = args[1]
                output = operation(y, x)
            case _:
                output = operation(self.get_args(1)[0])
        self.push(output)

    def run_ADD(self):
        self.push(self.pop() + self.pop())

    def run_SUB(self):
        self.push(-self.pop() + self.pop())

    def run_MUL(self):
        self.push(self.pop() * self.pop())

    def run_DIV(self):
        denom = self.pop()
        num = self.pop()
        self.push(num / denom)

    def run_POW(self):
        exponent = self.pop()
        base = self.pop()
        self.push(base**exponent)

    def run_KRON(self):
        op2 = self.pop()
        op1 = self.pop()
        self.push(qt.tensor(op1, op2))

    def run_NOT(self):
        self.push(not self.pop())

    def run_AND(self):
        self.push(self.pop() and self.pop())

    def run_OR(self):
        self.push(self.pop() or self.pop())

    def run_EQ(self):
        self.push(self.pop() == self.pop())

    def run_NEQ(self):
        self.push(self.pop() != self.pop())

    def run_LT(self):
        rhs = self.pop()
        lhs = self.pop()
        self.push(lhs < rhs)

    def run_LTEQ(self):
        rhs = self.pop()
        lhs = self.pop()
        self.push(lhs <= rhs)

    def run_GT(self):
        rhs = self.pop()
        lhs = self.pop()
        self.push(lhs > rhs)

    def run_GTEQ(self):
        rhs = self.pop()
        lhs = self.pop()
        self.push(lhs >= rhs)

    def run_INIT(self):
        targets = self.get_args(1)[0]
        for target in targets:
            if isinstance(target, QubitObject):
                target.state = qt.Qobj([1, 0])
            elif isinstance(target, ModeObject):
                target.state = qt.Qobj([self._fock_cutoff, 0])
            target.time = self.GLOBAL_T
        self.push(QutipVMNULL)

    def run_MEASURE(self):
        targets = self.get_args(1)[0]
        counts = {}
        for ind, target in enumerate(targets):
            probs = np.power(np.abs(target.state.full()), 2).squeeze()
            n_shots = self._n_shots
            inds = np.random.choice(len(probs), size=n_shots, p=probs)
            # print(inds)
            h_dims = 2
            if isinstance(target, ModeObject):
                h_dims = self._fock_cutoff
            opts = len(targets) * [list(range(h_dims))]
            bases = list(itertools.product(*opts))
            shots = np.array([bases[ind] for ind in inds])
            bitstrings = ["".join(map(str, shot)) for shot in shots]
            counts[ind] = {
                bitstring: bitstrings.count(bitstring) for bitstring in bitstrings
            }

        self.push(counts)

    def run_EXTRACT(self, name, index):
        if isinstance(name, ALIAS):
            self.push(RegisterObject(name=name.target, index=index))
        else:
            self.push(RegisterObject(name=name, index=index))

    def run_QREG(self, name, size):
        self.store[name] = [ListTerminators.LISTSTART]
        for n in range(size):
            obj = QubitObject()
            obj.register = RegisterObject(name=name, index=n)
            obj.time = self.GLOBAL_T
            obj.state = []
            self.registers[obj.register] = obj
            self.store[name].append(obj.register)
        self.store[name].append(ListTerminators.LISTEND)

    def run_MREG(self, name, size):
        self.store[name] = [ListTerminators.LISTSTART]
        for n in range(size):
            obj = ModeObject()
            obj.register = RegisterObject(name=name, index=n)
            obj.time = self.GLOBAL_T
            obj.state = []
            self.registers[obj.register] = obj
            self.store[name].append(obj.register)
        self.store[name].append(ListTerminators.LISTEND)

    # Pads the hamiltonian with additional dimensions if required and reorders states
    def _pad(self, hamiltonian, targets):
        states = []
        state_dims = 0
        for target in targets:
            states += [target.state]
            # print(f"target type: " + str(type(target)))
            if isinstance(target, QubitRegister):
                # print(target.qubits)
                state_dims += target.n
            else:
                # print(target.name)
                state_dims += 1
        h_dims = hamiltonian.dims[0]
        # print("state_dims are:")
        # print(state_dims)
        # print("h dims are:")
        # print(h_dims)
        diff = state_dims - len(h_dims)

        for _ in list(range(diff)):
            # h_dims[0] *= h_dims[1]
            hamiltonian = qt.tensor(qt.qeye(2), hamiltonian)

        states = qt.tensor(states)

        # print("Hamiltonian: ")
        # print(hamiltonian)
        # print("state: ")
        # print(states)

        return states, hamiltonian

    def run_EVOLVE(self):
        args = self.get_args(3)
        targets = args[0]
        # print(f"targets: " + str(targets))
        duration = args[1]
        # print(f"duration: " + str(duration))
        hamiltonian = args[2]
        # print(f"hamiltonian: " + str(hamiltonian))

        tspan = np.linspace(0, duration, round(duration / self._dt)).tolist()
        # results = {}

        if isinstance(hamiltonian, (MathExpr, OperatorExpr)):
            compiler_pass = Post(
                QutipQobjEvoGenerator(
                    fock_cutoff=self._fock_cutoff, current_time=self.GLOBAL_T
                )
            )
            hamiltonian = compiler_pass(hamiltonian)
            # print(f"hamiltonian after pass: " + str(hamiltonian))

        states, hamiltonian = self._pad(hamiltonian, targets)

        start_runtime = time.time()
        result_qobj = qt.sesolve(
            hamiltonian,
            states,  # Tensor product
            tspan,
            options={"store_states": True},
        )
        # print(self.results.runtime)
        elapsed_time = time.time() - start_runtime
        for target in targets:
            target.time += elapsed_time

        self.GLOBAL_T += duration
        # self.results.times.extend([t + self.results.times[-1] for t in tspan][1:])

        # for idx, key in enumerate(self.results.metrics.keys()):
        #     self.results.metrics[key].extend(result_qobj.expect[idx].tolist()[1:])

        # target.state = result_qobj.final_state
        # results = result_qobj.final_state.full().squeeze()

        qreg = QubitRegister()
        qreg.time = self.GLOBAL_T
        qreg.state = result_qobj.final_state
        for target in targets:
            if isinstance(target, QubitObject):
                if target.register not in qreg.register:
                    qreg.register.append(target.register)
            elif isinstance(target, QubitRegister):
                for register in target.register:
                    if register not in qreg.register:
                        qreg.register.append(register)

        for target in targets:
            if isinstance(target, QubitObject):
                self.registers[target.register] = qreg
            elif isinstance(target, QubitRegister):
                for register in target.register:
                    self.registers[register] = qreg

        self.push(QutipVMNULL)


class QutipInterpreter:
    def __init__(
        self,
        graph: ControlFlowGraph,
        codegen=None,
        n_shots: int = 10,
        fock_cutoff: int = 4,
        dt: float = 0.1,
    ):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.vm = QutipVM(n_shots, fock_cutoff, dt)
        self.INSTRUCTIONS = []
        self.codegen = codegen
        if codegen is None:
            self.codegen = QutipBackendInstructionsCodegen(fock_cutoff=fock_cutoff)

    def get_block(self, node: int = 0):
        return self.graph.blocks[node]

    def evaluate(self, stmt):
        instructions = self.codegen(stmt)
        # print(instructions)
        self.INSTRUCTIONS.append(instructions)
        self.vm.run(instructions)

    def run(self):
        node = 1
        current_block = self.get_block(node)

        while current_block.kind != "stop":
            stmt = current_block.stmt

            if current_block.kind == "branch":
                self.evaluate(stmt)
                cond = self.vm.pop()
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
                    # print(stmt)
                    self.evaluate(stmt)
                if current_block.succs:
                    current_block = current_block.succs[0]
                    continue
            # print(node)
            current_block = self.get_block(node)

        stack_top = self.vm.pop()
        if stack_top is None:
            return []
        return self.get_state(stack_top)

    def status(self):
        return self.vm.get_store()

    def get_state(self, return_values):
        return self.vm.get_state(return_values, verbose=True)

    def get_instructions(self):
        return self.INSTRUCTIONS
