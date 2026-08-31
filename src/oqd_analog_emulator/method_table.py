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
import typing
import warnings
from typing import Annotated, ClassVar, List, Literal, Union

import numpy as np
import qutip as qt
from oqd_compiler_infrastructure import Post, VisitableBaseModel
from oqd_core.interface.analog.expr import MathExpr, OperatorExpr
from pydantic import (
    BaseModel,
    Discriminator,
    TypeAdapter,
)

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
    
    def sort(self):
        sorted_name = sorted(self.name, key=lambda k: (k.name, k.index))
        permute_order = [self.name.index(x) for x in sorted_name]
        sorted_state = self.state.permute(permute_order)

        self.name = sorted_name
        self.state = sorted_state

        return self


AnalogVMNULL = [ListTerminators.LISTSTART, ListTerminators.LISTEND]


def recursive_filter(lst, cond):
    return list(
        map(
            lambda x: recursive_filter(x, cond) if isinstance(x, list) else x,
            filter(cond, lst)
        )
    )


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
        return QubitRegister(name=name, time_last_updated=self._GLOBAL_T, state=state, dims=dims)
        
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
            target.time_last_updated = self._GLOBAL_T

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

        if isinstance(hamiltonian, (MathExpr, OperatorExpr)):
            qobjevo_gen = Post(
                QutipQobjEvoGenerator(
                    fock_cutoff=self._fock_cutoff, current_time=self._GLOBAL_T
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

        self._GLOBAL_T += duration
        
        qreg = self._new_register(name=reordered_qubits, state=result_qobj.final_state, dims=len(reordered_qubits)**2)

        for target in reordered_qubits:
            registers[target] = qreg

        stack.push(AnalogVMNULL)


class DynamicsMixin:
    pass


class MethodTableBase(BaseModel):
        
    @classmethod
    def _is_classvar(cls, v):
        return v is ClassVar or typing.get_origin(v) is ClassVar

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for k, v in cls.__annotations__.items():
            if k == "class_":
                raise AttributeError("`class_` attribute should not be set manually.")

        cls.__annotations__["class_"] = Literal[cls.__name__]
        setattr(cls, "class_", cls.__name__)

        # Auto-register new method_table types
        MethodTableRegistry.register(cls)
    
    def get_state(self, return_values, stack, store, registers):
        if not isinstance(return_values, list):
            return return_values
        out = []
        for value in return_values:
            if isinstance(value, ListTerminators):
                continue
            if isinstance(value, list):
                out.append(self.get_state(value, stack, store, registers))
            elif isinstance(value, QubitName):
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


class MetaMethodTableRegistry(type):
    """
    Metaclass for the MethodTableRegistry
    """

    def __new__(cls, clsname, superclasses, attributedict):
        attributedict["method_tables"] = dict()
        return super().__new__(cls, clsname, superclasses, attributedict)

    def register(cls, method_table):
        """Registers a method_table into the MethodTableRegistry."""
        if not issubclass(method_table, MethodTableBase):
            raise TypeError("You may only register subclasses of MethodTableBase.")

        if method_table.__name__ in cls.method_tables.keys():
            warnings.warn(
                f"Overwriting previously registered `{method_table.__name__}` method_table of the same name.",
                UserWarning,
                stacklevel=2,
            )

        cls.method_tables[method_table.__name__] = method_table

    def clear(cls):
        """Clear all registered types (useful for testing)"""
        cls.method_tables.clear()

    @property
    def union(cls):
        """Get the current Union of all registered types"""

        if len(cls.method_tables) > 1:
            return Annotated[
                Union[tuple(cls.method_tables.values())], Discriminator(discriminator="class_")
            ]
        else:
            return next(iter(cls.method_tables.values()))

    @property
    def adapter(cls):
        """Get TypeAdapter for current registered types"""
        return TypeAdapter(cls.union)


class MethodTableRegistry(metaclass=MetaMethodTableRegistry):
    """
    Represents the MethodTableRegistry
    """

    pass


