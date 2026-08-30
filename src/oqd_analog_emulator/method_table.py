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
from typing import List

import numpy as np
import qutip as qt
from oqd_compiler_infrastructure import Post, VisitableBaseModel
from oqd_core.interface.analog.expr import MathExpr, OperatorExpr

from oqd_analog_emulator.instructions import ALIAS, ListTerminators
from oqd_analog_emulator.passes import QutipQobjEvoGenerator


class QubitName(VisitableBaseModel):
    name: str
    index: int

    def __hash__(self):
        return hash((self.name, self.index))


class QubitRegister(VisitableBaseModel):
    name: List[QubitName] = []
    time_last_updated: float
    state: object
    dims: int

    def __hash__(self):
        return hash(tuple(self.name))

    @property
    def n(self):
        return len(self.name)


AnalogVMNULL = [ListTerminators.LISTSTART, ListTerminators.LISTEND]


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
    def _new_register(self, name, state, dims):
        return QubitRegister(name=name, time_last_updated=self.GLOBAL_T, state=state, dims=dims)
        
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
        if isinstance(name, QubitName):
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
            target.state = qt.basis(target.dims, 0)
            target.time_last_updated = self.GLOBAL_T

        stack.push(AnalogVMNULL)

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
            h_dims = target.dims
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
            stack.push(QubitName(name=name.target, index=index))
        else:
            stack.push(QubitName(name=name, index=index))

    def run_QREG(self, name, size, stack, store, registers):
        store[name] = [ListTerminators.LISTSTART]
        for n in range(size):
            qubit = QubitName(name=name, index=n)
            obj = self._new_register(
                name=[qubit],
                dims=2,
                state=[],
            )
            registers[qubit] = obj
            store[name].append(qubit)
        store[name].append(ListTerminators.LISTEND)

    def run_MREG(self, name, size, stack, store, registers):
        store[name] = [ListTerminators.LISTSTART]
        for n in range(size):
            qubit = QubitName(name=name, index=n)
            obj = self._new_register(
                name=[qubit],
                dims=self._fock_cutoff,
                state=[],
            )
            registers[qubit] = obj
            store[name].append(qubit)
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
        # results = {}

        if isinstance(hamiltonian, (MathExpr, OperatorExpr)):
            compiler_pass = Post(
                QutipQobjEvoGenerator(
                    fock_cutoff=self._fock_cutoff, current_time=self.GLOBAL_T
                )
            )
            hamiltonian = compiler_pass(hamiltonian)

        states, padded_hamiltonian, reordered_qubits = self._pad(hamiltonian, targets)

        start_runtime = time.time()
        result_qobj = qt.sesolve(
            padded_hamiltonian,
            states,  # Tensor product
            tspan,
            options={"store_states": True},
        )

        elapsed_time = time.time() - start_runtime
        for target in reordered_qubits:
            registers[target].time_last_updated += elapsed_time

        self.GLOBAL_T += duration
        # self.results.times.extend([t + self.results.times[-1] for t in tspan][1:])

        # for idx, key in enumerate(self.results.metrics.keys()):
        #     self.results.metrics[key].extend(result_qobj.expect[idx].tolist()[1:])

        new_qubits = []
        for target in reordered_qubits:
            target_register = registers[target]
            for name in target_register.name:
                if name not in new_qubits:
                    new_qubits.append(name)
            # if isinstance(target, QubitObject):
            #     if target.name not in qreg.name:
            #         qreg.name.append(target.name)
            # elif isinstance(target, QubitRegister):
        
        qreg = self._new_register(name=new_qubits, state=result_qobj.final_state, dims=len(new_qubits)**2)

        for target in reordered_qubits:
            target_register = registers[target]
            for name in target_register.name:
                registers[name] = qreg

        # self.push(result_qobj.final_state.full().squeeze())
        stack.push(AnalogVMNULL)


class DynamicsMixin:
    pass



