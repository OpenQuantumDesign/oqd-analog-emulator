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

########################################################################################

import math
import warnings
from typing import Any, Generic, List, TypeVar

import numpy as np
import qutip as qt
from oqd_compiler_infrastructure import Post
from oqd_core.interface.analog.expr import MathExpr, OperatorExpr
from pydantic import BaseModel, ConfigDict

from oqd_analog_emulator.instructions import Alias, AnalogVMNULL, ListTerminators
from oqd_analog_emulator.passes import QutipQobjEvoGenerator

########################################################################################


class RegisterName(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    name: str
    index: int
    dim: int

    def __hash__(self):
        return hash((self.name, self.index))


class QuantumRegister(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    name: List[RegisterName] = []
    time: float
    time_last_updated: float
    state: Any

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


########################################################################################


def recursive_filter(lst, cond):
    return list(
        map(
            lambda x: recursive_filter(x, cond) if isinstance(x, list) else x,
            filter(cond, lst),
        )
    )


########################################################################################


class MethodTableOptionsBase(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )


M = TypeVar("MethodTableOptionsTypeVar", bound=MethodTableOptionsBase)


class MethodTableBase(Generic[M]):
    @classmethod
    def get_options_type(cls):
        return cls.__orig_bases__[0].__args__[0]

    def __init__(self, options: M | None = None, **kwargs):
        super().__init__()

        self.options = options if options else self.get_options_type()(**kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Auto-register new method_table types
        MethodTableRegistry.register(cls)

    def get_state(self, return_values, vm):
        if isinstance(return_values, RegisterName):
            return vm.registers[return_values]

        if not isinstance(return_values, list):
            return return_values

        out = []
        for value in return_values:
            if isinstance(value, ListTerminators):
                continue
            if isinstance(value, list):
                out.append(self.get_state(value, vm))
            elif isinstance(value, RegisterName):
                out.append((value, vm.registers[value]))
            else:
                out.append(value)
        return out

    def get_args(self, num, vm):
        out = []
        for _ in list(range(num)):
            item = vm.stack.pop()

            if isinstance(item, list):
                out.append(
                    recursive_filter(item, lambda x: not isinstance(x, ListTerminators))
                )
            else:
                out.append(item)
        return self.get_state(out, vm)

    def run(self, opcode, args, vm):
        getattr(self, f"run_{opcode}")(*args, vm)


########################################################################################


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

    def __getitem__(cls, idx):
        return cls.method_tables[idx]

    def get_options_type(cls, element):
        if isinstance(element, MethodTableBase):
            return element.get_options_type()

        if issubclass(element, MethodTableBase):
            return element.get_options_type()

        return cls.method_tables[element].get_options_type()


class MethodTableRegistry(metaclass=MetaMethodTableRegistry):
    """
    Represents the MethodTableRegistry
    """


########################################################################################


class ArithmeticMixin:
    def run_FUNC(self, func, vm):
        output = None
        operation = getattr(math, func, None)
        if operation is None:
            operation = getattr(np, func, None)
        if operation is None:
            raise ValueError("Unknown math function")

        match func:
            case "abs":
                output = abs(vm.stack.pop())
            case "heaviside":
                output = np.heaviside(vm.stack.pop(), 0)
            case "atan2":
                x = vm.stack.pop()
                y = vm.stack.pop()
                output = operation(y, x)
            case _:
                output = operation(vm.stack.pop())
        vm.stack.push(output)

    def run_ADD(self, vm):
        vm.stack.push(vm.stack.pop() + vm.stack.pop())

    def run_SUB(self, vm):
        vm.stack.push(-vm.stack.pop() + vm.stack.pop())

    def run_MUL(self, vm):
        vm.stack.push(vm.stack.pop() * vm.stack.pop())

    def run_DIV(self, vm):
        denom = vm.stack.pop()
        num = vm.stack.pop()
        vm.stack.push(num / denom)

    def run_POW(self, vm):
        exponent = vm.stack.pop()
        base = vm.stack.pop()
        vm.stack.push(base**exponent)


class BoolMixin:
    def run_NOT(self, vm):
        vm.stack.push(not vm.stack.pop())

    def run_AND(self, vm):
        vm.stack.push(vm.stack.pop() and vm.stack.pop())

    def run_OR(self, vm):
        vm.stack.push(vm.stack.pop() or vm.stack.pop())

    def run_EQ(self, vm):
        vm.stack.push(vm.stack.pop() == vm.stack.pop())

    def run_NEQ(self, vm):
        vm.stack.push(vm.stack.pop() != vm.stack.pop())

    def run_LT(self, vm):
        rhs = vm.stack.pop()
        lhs = vm.stack.pop()
        vm.stack.push(lhs < rhs)

    def run_LTEQ(self, vm):
        rhs = vm.stack.pop()
        lhs = vm.stack.pop()
        vm.stack.push(lhs <= rhs)

    def run_GT(self, vm):
        rhs = vm.stack.pop()
        lhs = vm.stack.pop()
        vm.stack.push(lhs > rhs)

    def run_GTEQ(self, vm):
        rhs = vm.stack.pop()
        lhs = vm.stack.pop()
        vm.stack.push(lhs >= rhs)


class StackStoreMixin:
    def run_GLOBAL(self, name, vm):
        if name not in vm.store:
            vm.store[name] = None

    def run_CONST(self, value, vm):
        vm.stack.push(value)

    def run_STORE(self, name, vm):
        vm.store[name] = vm.stack.pop()

    def run_LOAD(self, name, vm):
        while isinstance(vm.store.get(name, None), Alias):
            name = vm.store[name].target

        if name in vm.store:
            item = vm.store[name]
            vm.stack.push(item)
        else:
            raise ValueError

    def run_EXTRACT(self, name, index, vm):
        while isinstance(vm.store.get(name, None), Alias):
            name = vm.store[name].target

        if name in vm.store and index < len(vm.store[name]) - 2:
            item = vm.store[name][index + 1]
            vm.stack.push(item)
        else:
            raise ValueError


class QutipMixin:
    def _new_register(self, name, state, vm):
        return QuantumRegister(
            name=name,
            time=vm.machine_time,
            time_last_updated=vm.machine_time,
            state=state,
        )

    def run_QREG(self, name, size, dim, vm):
        if vm.registers.contains_name(name):
            vm.registers.wipe(name)

        store_value = [ListTerminators.LISTSTART]
        for n in range(size):
            qubit = RegisterName(name=name, index=n, dim=dim)
            reg = self._new_register(name=[qubit], state=None, vm=vm)
            vm.registers[qubit] = reg
            store_value.append(qubit)
        store_value.append(ListTerminators.LISTEND)

        vm.store[name] = store_value

    def run_MREG(self, name, size, vm):
        self.run_QREG(name, size, self.options.fock_cutoff, vm)

    def run_KRON(self, vm):
        op2 = vm.stack.pop()
        op1 = vm.stack.pop()
        vm.stack.push(qt.tensor(op1, op2))

    def run_INIT(self, vm):
        targets = self.get_args(1, vm)[0]

        targets = targets if isinstance(targets, list) else [targets]

        for name, target in targets:
            vm.registers[name].state = qt.basis(
                np.prod([n.dim for n in vm.registers[name].name]), 0
            )
            vm.registers[name].time_last_updated = vm.machine_time

        vm.stack.push(AnalogVMNULL)

    def run_MEASURE(self, vm):
        # TODO: Fix measurements
        # targets = self.get_args(1, vm)[0]
        # if not isinstance(targets, list):
        #     targets = [targets]

        # qubits = []
        # actual_qubits = []
        # for target, name in targets:
        #     qubits.append(target)
        #     actual_qubits.append(name)
        # targets = actual_qubits
        # counts = {}
        # for ind, target in enumerate(targets):
        #     probs = np.power(np.abs(target.state.full()), 2).squeeze()
        #     n_shots = self._n_shots
        #     inds = np.random.choice(len(probs), size=n_shots, p=probs)
        #     h_dims = target.dims
        #     opts = len(targets) * [list(range(h_dims))]
        #     bases = list(itertools.product(*opts))
        #     shots = np.array([bases[ind] for ind in inds])
        #     bitstrings = ["".join(map(str, shot)) for shot in shots]
        #     counts[ind] = {
        #         bitstring: bitstrings.count(bitstring) for bitstring in bitstrings
        #     }

        vm.stack.push(AnalogVMNULL)

        # Pads the hamiltonian with additional dimensions if required and reorders states

    def _pad_hamiltonian(self, hamiltonian, targets):
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

    def run_EVOLVE(self, vm):
        args = self.get_args(3, vm)
        targets = args[0]

        duration = args[1]
        hamiltonian = args[2]

        tspan = np.arange(0, duration, self.options.dt)
        if tspan[-1] != duration:
            tspan = np.concat([tspan, [duration]])
        tspan += vm.machine_time

        if isinstance(hamiltonian, (MathExpr, OperatorExpr)):
            qobjevo_gen = Post(
                QutipQobjEvoGenerator(
                    fock_cutoff=self.options.fock_cutoff, current_time=vm.machine_time
                )
            )
            hamiltonian = qobjevo_gen(hamiltonian)

        states, padded_hamiltonian, reordered_qubits = self._pad_hamiltonian(
            hamiltonian, targets
        )

        result_qobj = qt.sesolve(
            padded_hamiltonian,
            states,  # Tensor product
            tspan,
            options={"store_states": True},
        )

        vm.machine_time += duration

        qreg = self._new_register(
            name=reordered_qubits,
            state=result_qobj.final_state,
            vm=vm,
        ).sort()

        for target in reordered_qubits:
            vm.registers[target] = qreg

        for reg in vm.registers.keys():
            vm.registers[reg].time = vm.machine_time

        vm.stack.push(AnalogVMNULL)
