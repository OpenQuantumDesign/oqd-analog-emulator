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
from typing import List

import numpy as np
import qutip as qt
from oqd_compiler_infrastructure import Post, VisitableBaseModel
from oqd_core.analysis.utils import ControlFlowGraph
from oqd_core.interface.analog.expr import MathExpr, OperatorExpr

from oqd_analog_emulator.instructions import (
    ALIAS,
    ListTerminators,
    QutipBackendInstructions,
    QutipBackendInstructionsCodegen,
)
from oqd_analog_emulator.passes import QutipQobjEvoGenerator

########################################################################################


class RegisterObject(VisitableBaseModel):
    name: str
    index: int

    def __hash__(self):
        return hash((self.name, self.index))


class QubitObject(VisitableBaseModel):
    name: RegisterObject
    time: float
    state: object

    def __hash__(self):
        return hash(self.name)


class QubitRegister(VisitableBaseModel):
    name: List[RegisterObject] = []
    time: float
    state: object

    def __hash__(self):
        return hash(tuple(self.name))

    @property
    def n(self):
        return len(self.name)

    def sort(self):
        sorted_name = sorted(self.name, key=lambda k: (k.name, k.index))
        permute_order = [self.name.index(x) for x in sorted_name]
        sorted_state = self.state.permute(permute_order)

        self.name = sorted_name
        self.state = sorted_state

        return self


class ModeObject(VisitableBaseModel):
    name: RegisterObject
    time: float
    state: object

    def __hash__(self):
        return hash(self.name)


class ModeRegister(VisitableBaseModel):
    name: List[RegisterObject] = []
    time: float
    state: object

    def __hash__(self):
        return hash(tuple(self.name))

    @property
    def n(self):
        return len(self.name)


QutipVMNULL = [ListTerminators.LISTSTART, ListTerminators.LISTEND]


########################################################################################


def recursive_filter(lst, cond):
    return list(
        map(
            lambda x: recursive_filter(x, cond) if isinstance(x, list) else x,
            filter(cond, lst),
        )
    )


########################################################################################


class ArithmeticMixin:
    def run_FUNC(self, func, stack, store, registers):
        output = None
        operation = getattr(math, func, None)
        if operation is None:
            operation = getattr(np, func, None)
        if operation is None:
            raise ValueError("Unknown math function")

        match func:
            case "abs":
                output = abs(stack.pop())
            case "heaviside":
                output = np.heaviside(stack.pop(), 0)
            case "atan2":
                x = stack.pop()
                y = stack.pop()
                output = operation(y, x)
            case _:
                output = operation(stack.pop())
        stack.push(output)

    def run_ADD(self, stack, store, registers):
        stack.push(stack.pop() + stack.pop())

    def run_SUB(self, stack, store, registers):
        stack.push(-stack.pop() + stack.pop())

    def run_MUL(self, stack, store, registers):
        stack.push(stack.pop() * stack.pop())

    def run_DIV(self, stack, store, registers):
        denom = stack.pop()
        num = stack.pop()
        stack.push(num / denom)

    def run_POW(self, stack, store, registers):
        exponent = stack.pop()
        base = stack.pop()
        stack.push(base**exponent)


class BoolMixin:
    def run_NOT(self, stack, store, registers):
        stack.push(not stack.pop())

    def run_AND(self, stack, store, registers):
        stack.push(stack.pop() and stack.pop())

    def run_OR(self, stack, store, registers):
        stack.push(stack.pop() or stack.pop())

    def run_EQ(self, stack, store, registers):
        stack.push(stack.pop() == stack.pop())

    def run_NEQ(self, stack, store, registers):
        stack.push(stack.pop() != stack.pop())

    def run_LT(self, stack, store, registers):
        rhs = stack.pop()
        lhs = stack.pop()
        stack.push(lhs < rhs)

    def run_LTEQ(self, stack, store, registers):
        rhs = stack.pop()
        lhs = stack.pop()
        stack.push(lhs <= rhs)

    def run_GT(self, stack, store, registers):
        rhs = stack.pop()
        lhs = stack.pop()
        stack.push(lhs > rhs)

    def run_GTEQ(self, stack, store, registers):
        rhs = stack.pop()
        lhs = stack.pop()
        stack.push(lhs >= rhs)


class QutipMixin:
    def _new_register(self, name, state):
        return QubitRegister(name=name, time=self.GLOBAL_T, state=state)

    def run_GLOBAL(self, name, stack, store, registers):
        if name not in store:
            store[name] = None

    def run_CONST(self, value, stack, store, registers):
        stack.push(value)

    def run_STORE(self, name, stack, store, registers):
        store[name] = stack.pop()

    def run_LOAD(self, name, stack, store, registers):
        if isinstance(name, ALIAS):
            self.run_LOAD(name.target, stack, store, registers)
        if isinstance(name, RegisterObject):
            stack.push(registers[name])
        if name in store:
            item = store[name]
            stack.push(item)
        else:
            raise ValueError

    def run_KRON(self, stack, store, registers):
        op2 = stack.pop()
        op1 = stack.pop()
        stack.push(qt.tensor(op1, op2))

    def run_INIT(self, stack, store, registers):
        targets = self.get_args(1, stack, store, registers)[0]
        qubits = []
        actual_qubits = []
        if not isinstance(targets, list):
            targets = [targets]
        for target, name in targets:
            qubits.append(target)
            actual_qubits.append(name)
        targets = actual_qubits

        for target in targets:
            if isinstance(target, QubitObject):
                target.state = qt.basis(2, 0)
            elif isinstance(target, ModeObject):
                target.state = qt.basis(self._fock_cutoff, 0)
            target.time = self.GLOBAL_T

        stack.push(QutipVMNULL)

    def run_MEASURE(self, stack, store, registers):
        targets = self.get_args(1, stack, store, registers)[0]
        if not isinstance(targets, list):
            targets = [targets]

        qubits = []
        actual_qubits = []
        for target, name in targets:
            qubits.append(target)
            actual_qubits.append(name)
        targets = actual_qubits
        counts = {}
        for ind, target in enumerate(targets):
            probs = np.power(np.abs(target.state.full()), 2).squeeze()
            n_shots = self._n_shots
            inds = np.random.choice(len(probs), size=n_shots, p=probs)
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

        stack.push(counts)

    def run_EXTRACT(self, name, index, stack, store, registers):
        if isinstance(name, ALIAS):
            stack.push(RegisterObject(name=name.target, index=index))
        else:
            stack.push(RegisterObject(name=name, index=index))

    def run_QREG(self, name, size, stack, store, registers):
        store[name] = [ListTerminators.LISTSTART]
        for n in range(size):
            obj = QubitObject(
                name=RegisterObject(name=name, index=n),
                time=self.GLOBAL_T,
                state=[],
            )
            registers[obj.name] = obj
            store[name].append(obj.name)
        store[name].append(ListTerminators.LISTEND)

    def run_MREG(self, name, size, stack, store, registers):
        store[name] = [ListTerminators.LISTSTART]
        for n in range(size):
            obj = ModeObject(
                name=RegisterObject(name=name, index=n),
                time=self.GLOBAL_T,
                state=[],
            )
            registers[obj.name] = obj
            store[name].append(obj.name)
        store[name].append(ListTerminators.LISTEND)

        # Pads the hamiltonian with additional dimensions if required and reorders states

    def _pad(self, hamiltonian, targets):
        qubits, targets = (
            zip(*targets) if isinstance(targets, list) else zip(*[targets])
        )
        targets, qubits = list(targets), list(qubits)

        _targets = map(
            lambda x: (x.name, x.state),
            set(targets),
        )

        all_qubits = []
        states = []
        for q, s in _targets:
            states.append(s)

            if isinstance(q, list):
                all_qubits.extend(q)
            else:
                all_qubits.append(q)

        h_dims = hamiltonian.dims[0]
        diff = len(all_qubits) - len(h_dims)

        padded_hamiltonian = qt.tensor(
            *[qt.qeye(2) for _ in list(range(diff))], hamiltonian
        )

        # Calculate State
        padded_qubits = [*set(all_qubits).difference(qubits), *qubits]
        permute_order = [all_qubits.index(x) for x in padded_qubits]

        states = qt.tensor(*states)
        states = states.permute(permute_order)

        return states, padded_hamiltonian, padded_qubits

    def run_EVOLVE(self, stack, store, registers):
        args = self.get_args(3, stack, store, registers)
        targets = args[0]
        duration = args[1]
        hamiltonian = args[2]

        tspan = np.linspace(0, duration, round(duration / self._dt)).tolist()

        if isinstance(hamiltonian, (MathExpr, OperatorExpr)):
            qobjevo_gen = Post(
                QutipQobjEvoGenerator(
                    fock_cutoff=self._fock_cutoff, current_time=self.GLOBAL_T
                )
            )
            hamiltonian = qobjevo_gen(hamiltonian)

        states, padded_hamiltonian, reordered_qubits = self._pad(hamiltonian, targets)

        result_qobj = qt.sesolve(
            padded_hamiltonian,
            states,  # Tensor product
            tspan,
            options={"store_states": True},
        )

        self.GLOBAL_T += duration
        # self.results.times.extend([t + self.results.times[-1] for t in tspan][1:])

        # for idx, key in enumerate(self.results.metrics.keys()):
        #     self.results.metrics[key].extend(result_qobj.expect[idx].tolist()[1:])

        qreg = self._new_register(reordered_qubits, result_qobj.final_state)
        for target in reordered_qubits:
            registers[target] = qreg

        stack.push(QutipVMNULL)


class DynamicsMixin:
    pass


########################################################################################


class QutipMethodTable(ArithmeticMixin, BoolMixin, QutipMixin):
    def __init__(self, n_shots=10, fock_cutoff=4, dt=0.1):
        self._n_shots = n_shots
        self._fock_cutoff = fock_cutoff
        self._dt = dt
        self.GLOBAL_T = 0.0

    def get_state(self, return_values, stack, store, registers):
        if not isinstance(return_values, list):
            return return_values
        out = []
        for value in return_values:
            if isinstance(value, ListTerminators):
                continue
            if isinstance(value, list):
                out.append(self.get_state(value, stack, store, registers))
            elif isinstance(value, RegisterObject):
                out.append((value, registers[value]))
            else:
                out.append(value)
        return out

    def get_args(self, num, stack, store, registers):
        out = []
        for _ in list(range(num)):
            item = stack.pop()
            if isinstance(item, list):
                out.append(
                    recursive_filter(item, lambda x: not isinstance(x, ListTerminators))
                )
            else:
                out.append(item)
        return self.get_state(out, stack, store, registers)

    def run(self, opcode, args, stack, store, registers):
        getattr(self, f"run_{opcode}")(*args, stack, store, registers)


########################################################################################
class QutipVMStack(VisitableBaseModel):
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


########################################################################################


class QutipVM:
    def __init__(self, n_shots=10, fock_cutoff=4, dt=0.1):
        self.stack = QutipVMStack()
        self.store = {}
        self.registers = {}

        self.history = {}
        self.method_table = QutipMethodTable(
            n_shots=n_shots, fock_cutoff=fock_cutoff, dt=dt
        )

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


########################################################################################


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
