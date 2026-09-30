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

from __future__ import annotations

import operator
import warnings
from typing import Any, Generic, List, TypeVar

import numpy as np
import qutip as qt
from pydantic import BaseModel, ConfigDict
from scipy.sparse import csr_matrix

from oqd_analog_emulator.instructions import AnalogVMNULL, ListTerminators

########################################################################################


class QuantumRegisterPointer(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    dim: int

    def __hash__(self):
        return self.index


class QuantumRegister(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    parts: List[QuantumRegisterPointer] = []
    time: float
    time_last_updated: float
    state: Any = None

    def __hash__(self):
        return hash(tuple(sorted(self.parts, key=lambda x: x.index)))

    def __len__(self):
        return len(self.parts)

    def sort(self):
        sorted_parts = sorted(self.parts, key=lambda x: x.index)
        self.permute(sorted_parts)

        return self

    def permute(self, order: List[int] | List[QuantumRegister]):
        match order:
            case list() if all([isinstance(i, int) for i in order]):
                pass
            case list() if all([isinstance(i, QuantumRegisterPointer) for i in order]):
                order = [self.parts.index(i) for i in order]
            case _:
                raise ValueError("Unsupported order for permute in QuantumRegister")

        self.parts = [self.parts[i] for i in order]
        self.state = self.state.permute(order)
        return self


########################################################################################


def recursive_filter(lst, cond):
    return list(
        map(
            lambda x: recursive_filter(x, cond) if isinstance(x, list) else x,
            filter(cond, lst),
        )
    )


def _build_callable(func, *args):
    match args:
        case (int() | float() | complex() | qt.Qobj(),) as operand:
            return func(args[0])

        case (operand,) if callable(operand):
            return lambda t, s: func(args[0](t, s))

        case (
            int() | float() | complex() | qt.Qobj() as left,
            int() | float() | complex() | qt.Qobj() as right,
        ):
            return func(left, right)

        case (left, int() | float() | complex() | qt.Qobj() as right) if callable(left):
            return lambda t, s: func(left(t, s), right)

        case (int() | float() | complex() | qt.Qobj() as left, right) if callable(
            right
        ):
            return lambda t, s: func(left, right(t, s))

        case (left, right) if callable(left) and callable(right):
            return lambda t, s: func(left(t, s), right(t, s))

        case _:
            raise ValueError()


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
        if isinstance(return_values, QuantumRegisterPointer):
            return vm.registers[return_values]

        if not isinstance(return_values, list):
            return return_values

        out = []
        for value in return_values:
            if isinstance(value, ListTerminators):
                continue
            if isinstance(value, list):
                out.append(self.get_state(value, vm))
            elif isinstance(value, QuantumRegisterPointer):
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


class PrintMixin:
    def run_PRINT(self, vm):
        arg = self.get_args(num=1, vm=vm)[0]

        print(arg)


class ArithmeticMixin:
    def run_NEG(self, vm):
        arg = self.get_args(num=1, vm=vm)[0]

        vm.stack.push(_build_callable(operator.neg, arg))

    def run_POS(self, vm):
        arg = self.get_args(num=1, vm=vm)[0]

        vm.stack.push(_build_callable(operator.pos, arg))

    def run_ADD(self, vm):
        args = left, right = self.get_args(num=2, vm=vm)

        vm.stack.push(_build_callable(operator.add, *args))

    def run_SUB(self, vm):
        args = left, right = self.get_args(num=2, vm=vm)

        vm.stack.push(_build_callable(operator.sub, *args))

    def run_MUL(self, vm):
        args = left, right = self.get_args(num=2, vm=vm)

        vm.stack.push(_build_callable(operator.mul, *args))

    def run_DIV(self, vm):
        args = left, right = self.get_args(num=2, vm=vm)

        vm.stack.push(_build_callable(operator.truediv, *args))

    def run_POW(self, vm):
        args = left, right = self.get_args(num=2, vm=vm)

        vm.stack.push(_build_callable(operator.pow, *args))

    def run_MFUNC(self, vm):
        name = self.get_args(num=1, vm=vm)[0]

        match name:
            case "$atan2":
                args = self.get_args(num=2, vm=vm)
            case _:
                args = self.get_args(num=1, vm=vm)

        vm.stack.push(_build_callable(vm.store[name], *args))


class FunctionMixin:
    def run_LEN(self, vm):
        arg = self.get_args(num=1, vm=vm)[0]

        vm.stack.push(len(arg))

    def run_FLATTEN(self, vm):
        arg = self.get_args(num=1, vm=vm)[0]

        vm.stack.push(
            [ListTerminators.LISTSTART]
            + [e for sub in arg for e in sub if not isinstance(e, ListTerminators)]
            + [ListTerminators.LISTEND]
        )

    def run_RANGE(self, vm):
        args = self.get_args(num=3, vm=vm)

        vm.stack.push(
            [ListTerminators.LISTSTART]
            + [i for i in range(*args)]
            + [ListTerminators.LISTEND]
        )


class BoolMixin:
    def run_NOT(self, vm):
        vm.stack.push(not vm.stack.pop())

    def run_AND(self, vm):
        vm.stack.push(vm.stack.pop() and vm.stack.pop())

    def run_OR(self, vm):
        vm.stack.push(vm.stack.pop() or vm.stack.pop())

    def run_XOR(self, vm):
        vm.stack.push(vm.stack.pop() ^ vm.stack.pop())

    def run_EQ(self, vm):
        vm.stack.push(vm.stack.pop() == vm.stack.pop())

    def run_NEQ(self, vm):
        vm.stack.push(vm.stack.pop() != vm.stack.pop())

    def run_LT(self, vm):
        vm.stack.push(vm.stack.pop() < vm.stack.pop())

    def run_LEQ(self, vm):
        vm.stack.push(vm.stack.pop() <= vm.stack.pop())

    def run_GT(self, vm):
        vm.stack.push(vm.stack.pop() > vm.stack.pop())

    def run_GEQ(self, vm):
        vm.stack.push(vm.stack.pop() >= vm.stack.pop())


class StackStoreMixin:
    def run_GLOBAL(self, vm):
        name = vm.stack.pop()
        if name not in vm.store:
            vm.store[name] = None

    def run_CONST(self, value, vm):
        vm.stack.push(value)

    def run_STORE(self, vm):
        name = vm.stack.pop()

        vm.store[name] = vm.stack.pop()

    def run_LOAD(self, vm):
        name = vm.stack.pop()
        while True:
            value = vm.store.get(name, None)

            if not isinstance(value, str) or not value.startswith("&"):
                break

            name = value.removeprefix("&")

        vm.stack.push(vm.store[name])

    def run_LOADV(self, vm):
        name = vm.stack.pop()

        value = vm.store.get(name, None)

        if value is None:
            raise ValueError("")

        vm.stack.push(value)

    def run_EXTRACT(self, vm):
        value = vm.stack.pop()
        index = vm.stack.pop()

        if index < len(value) - 2:
            item = value[index + 1]
            vm.stack.push(item)
        else:
            raise ValueError


class QutipMixin:
    def _new_register(self, parts, state, vm):
        return QuantumRegister(
            parts=parts,
            time=vm.machine_time,
            time_last_updated=vm.machine_time,
            state=state,
        )

    def run_PI(self, vm):
        level1, level2, dim = self.get_args(num=3, vm=vm)

        data = np.ones(2, dtype=np.complex64)
        col = np.array([level1, level2])
        row = np.array([level1, level2])

        op = csr_matrix((data, (row, col)), shape=(dim, dim))
        vm.stack.push(qt.Qobj(op))

    def run_PX(self, vm):
        level1, level2, dim = self.get_args(num=3, vm=vm)

        data = np.ones(2, dtype=np.complex64)
        col = np.array([level1, level2])
        row = np.array([level2, level1])

        op = csr_matrix((data, (row, col)), shape=(dim, dim))
        vm.stack.push(qt.Qobj(op))

    def run_PY(self, vm):
        level1, level2, dim = self.get_args(num=3, vm=vm)

        data = np.array([1j, -1j], dtype=np.complex64)
        col = np.array([level1, level2])
        row = np.array([level2, level1])

        op = csr_matrix((data, (row, col)), shape=(dim, dim))
        vm.stack.push(qt.Qobj(op))

    def run_PZ(self, vm):
        level1, level2, dim = self.get_args(num=3, vm=vm)

        data = np.array([1, -1], dtype=np.complex64)
        col = np.array([level1, level2])
        row = np.array([level1, level2])

        op = csr_matrix((data, (row, col)), shape=(dim, dim))
        vm.stack.push(qt.Qobj(op))

    def run_MI(self, vm):
        dim = self.options.fock_cutoff

        vm.stack.push(qt.qeye(dim, dtype=qt.data.CSR))

    def run_MA(self, vm):
        dim = self.options.fock_cutoff

        vm.stack.push(qt.destroy(dim, dtype=qt.data.CSR))

    def run_MC(self, vm):
        dim = self.options.fock_cutoff

        vm.stack.push(qt.create(dim, dtype=qt.data.CSR))

    def run_KRON(self, vm):
        args = self.get_args(2, vm)
        vm.stack.push(_build_callable(qt.tensor, *args))

    def run_QREG(self, vm):
        size, dim = self.get_args(num=2, vm=vm)
        names = vm.registers.create(size, dim, vm.machine_time)
        vm.stack.push([ListTerminators.LISTSTART, *names, ListTerminators.LISTEND])

    def run_MREG(self, vm):
        size = self.get_args(num=1, vm=vm)[0]
        names = vm.registers.create(size, self.options.fock_cutoff, vm.machine_time)
        vm.stack.push([ListTerminators.LISTSTART, *names, ListTerminators.LISTEND])

        # Pads the hamiltonian with additional dimensions if required and reorders states

    def _pad_qops(self, qops, targets):
        # unpack parts and unique registers
        parts, regs = zip(*targets) if isinstance(targets, list) else zip(*[targets])
        parts = list(parts)
        regs = set(regs)

        # extract values from registers
        regs = tuple((x.parts, x.state) for x in regs)

        # get all parts in registers
        total_states = [state for (_, state) in regs]
        total_parts = [part for (parts, _) in regs for part in parts]

        # get all parts not in operators
        remaining_parts = [part for part in total_parts if part not in parts]

        # compute padded operators
        padded_qops = [
            qt.tensor(*[qt.qeye(part.dim) for part in remaining_parts], qop)
            for qop in qops
        ]

        # Calculate permutation to take total_parts to padded_parts
        padded_parts = remaining_parts + parts
        permute_order = [total_parts.index(x) for x in padded_parts]

        # Turn all states to dm if any state is dm
        total_states = (
            total_states
            if all(list(map(lambda s: s.isket, total_states)))
            else [qt.ket2dm(s) if s.isket else s for s in total_states]
        )

        # Compute total_state
        total_state = qt.tensor(*total_states)
        total_state = total_state.permute(permute_order)

        return total_state, padded_qops, padded_parts

    def _verify_normalized(self, t, state):
        if not np.isclose(state.norm(), 1.0, atol=0, rtol=self.options.normalized_tol):
            raise ValueError("State not normalized through evolution")
        return

    def run_EVOLVE(self, vm):
        args = self.get_args(4, vm)
        hamiltonian = args[0]
        jumps = args[1]
        duration = args[2]
        targets = args[3]
        targets = targets if isinstance(targets, list) else [targets]

        for ptr, _ in targets:
            if vm.registers[ptr].state is None:
                raise ValueError("Attempted to evolve uninitialized qubit")

        tspan = np.arange(0, duration, self.options.dt)
        if tspan[-1] != duration:
            tspan = np.concat([tspan, [duration]])
        tspan += vm.machine_time

        H = (
            hamiltonian
            if isinstance(hamiltonian, qt.Qobj)
            else qt.QobjEvo(lambda t: hamiltonian(t, t - vm.machine_time))
        )
        Ls = [
            jump
            if isinstance(jump, qt.Qobj)
            else qt.QobjEvo(lambda t: jump(t, t - vm.machine_time))
            for jump in jumps
        ]

        if self.options.ignore_jumps or Ls == []:
            state, ops, reordered_parts = self._pad_qops([H], targets)

            H = ops[0]

            evo_res = qt.sesolve(
                H,
                state,
                tspan,
                e_ops={"_verify_normalized": self._verify_normalized}
                if self.options.verify_normalized
                else {},
                options=self.options.solver_options,
            )
        else:
            state, ops, reordered_parts = self._pad_qops([H, *Ls], targets)

            H = ops[0]
            Ls = ops[1:]

            evo_res = qt.mesolve(
                H,
                state,
                tspan,
                Ls,
                e_ops={"_verify_normalized": self._verify_normalized}
                if self.options.verify_normalized
                else {},
                options=self.options.solver_options,
            )

        vm.machine_time += duration

        qreg = self._new_register(
            parts=reordered_parts,
            state=evo_res.final_state,
            vm=vm,
        ).sort()

        for ptr in reordered_parts:
            vm.registers[ptr] = qreg

        for ptr in vm.registers.keys():
            vm.registers[ptr].time = vm.machine_time

        vm.stack.push(AnalogVMNULL)

    def run_INIT(self, vm):
        if self.options.singleshot_init:
            self._run_singleshot_INIT(vm)
        else:
            self._run_ensemble_INIT(vm)

    def run_MEASURE(self, vm):
        if self.options.ignore_measurements:
            self._run_noop_MEASURE(vm)
        else:
            self._run_singleshot_MEASURE(vm)

    def _run_ensemble_INIT(self, vm):
        targets = self.get_args(1, vm)[0]

        targets = targets if isinstance(targets, list) else [targets]
        targets = [part for part, target in targets]

        while targets:
            target = targets[0]

            current_state = vm.registers[target].state

            if current_state is None:
                vm.registers[target] = self._new_register(
                    parts=vm.registers[target].parts,
                    state=qt.basis(
                        np.prod([t.dim for t in vm.registers[target].parts]),
                        0,
                        dtype=qt.data.CSR,
                    ),
                    vm=vm,
                )
                continue

            system = vm.registers[target].parts
            initialized_subsystem = [
                i for i in range(len(system)) if system[i] in targets
            ]
            remaining_subsystem = [
                i for i in range(len(system)) if system[i] not in targets
            ]

            for ptr in (system[i] for i in initialized_subsystem):
                vm.registers[ptr] = self._new_register(
                    parts=[ptr], state=qt.basis(ptr.dim, 0, dtype=qt.data.CSR), vm=vm
                )

            new_state = current_state.ptrace(remaining_subsystem)

            for ptr in (system[i] for i in remaining_subsystem):
                vm.registers[ptr] = self._new_register(
                    parts=[system[i] for i in remaining_subsystem],
                    state=new_state,
                    vm=vm,
                ).sort()

            targets = [t for t in targets if t not in system]

        vm.stack.push(AnalogVMNULL)

    def _run_noop_MEASURE(self, vm):
        self.get_args(num=1, vm=vm)
        warnings.warn("Measurements are being ignored by the method table")
        vm.stack.push(AnalogVMNULL)

    def _projective_measure_single(self, reg, target, vm):
        remainder = [i for i in reg.parts if i != target]

        reg = reg.permute([target] + remainder)

        ops = [qt.basis(target.dim, i, dtype=qt.data.CSR) for i in range(target.dim)]
        ops = [
            qt.tensor(
                op.proj(),
                *[qt.qeye(i.dim, dtype=qt.data.CSR) for i in remainder],
            )
            for op in ops
        ]

        outcome, new_state = qt.measurement.measure(reg.state, ops)

        target_state = qt.basis(target.dim, outcome, dtype=qt.data.CSR)

        if remainder == []:
            return outcome, target_state, None

        remainder_state = (
            qt.tensor(
                target_state.dag(),
                *[qt.qeye(i.dim, dtype=qt.data.CSR) for i in remainder],
            )
            * new_state
        )
        remainder_state.dims = [remainder_state.dims[0][1:], remainder_state.dims[1]]

        remainder_reg = self._new_register(
            parts=remainder, state=remainder_state, vm=vm
        )

        return outcome, target_state, remainder_reg

    def _run_singleshot_MEASURE(self, vm):
        targets = self.get_args(1, vm)[0]

        targets = targets if isinstance(targets, list) else [targets]
        targets = [ptr for ptr, target in targets]

        _targets = targets

        outcomes = []
        outcome_order = []
        while _targets:
            target = _targets[0]

            reg = vm.registers[target]

            if reg.state is None:
                raise ValueError("Attempted to measure uninitialized qubit")

            # Calculate measured subsystem and remaining subsystem
            system = vm.registers[target].parts
            measured_subsystem = [i for i in system if i in _targets]
            remaining_subsystem = [i for i in system if i not in _targets]

            # Calculate new permutation of current state
            reg = reg.permute(measured_subsystem + remaining_subsystem)

            while measured_subsystem:
                ptr = measured_subsystem.pop(0)

                outcome, target_state, reg = self._projective_measure_single(
                    reg, ptr, vm
                )

                vm.registers[ptr] = self._new_register(
                    parts=[ptr], state=target_state, vm=vm
                )

                outcomes.append(outcome)
                outcome_order.append(ptr)

            for ptr in remaining_subsystem:
                vm.registers[ptr] = reg.sort()

            _targets = [t for t in _targets if t not in system]

        reordered_outcomes = [
            outcomes[i] for i in [outcome_order.index(j) for j in targets]
        ]

        vm.stack.push(
            [ListTerminators.LISTSTART, *reordered_outcomes, ListTerminators.LISTEND]
        )

    def _run_singleshot_INIT(self, vm):
        targets = self.get_args(1, vm)[0]

        targets = targets if isinstance(targets, list) else [targets]
        targets = [ptr for ptr, target in targets]

        _targets = targets

        while _targets:
            target = _targets[0]

            reg = vm.registers[target]

            if reg.state is None:
                vm.registers[target] = self._new_register(
                    parts=vm.registers[target].parts,
                    state=qt.basis(
                        np.prod([ptr.dim for ptr in vm.registers[target].parts]),
                        0,
                        dtype=qt.data.CSR,
                    ),
                    vm=vm,
                )
                continue

            # Calculate measured subsystem and remaining subsystem
            system = vm.registers[target].parts
            measured_subsystem = [i for i in system if i in _targets]
            remaining_subsystem = [i for i in system if i not in _targets]

            # Calculate new permutation of current state
            reg = reg.permute(measured_subsystem + remaining_subsystem)

            while measured_subsystem:
                ptr = measured_subsystem.pop(0)

                _, _, reg = self._projective_measure_single(reg, ptr, vm)

                vm.registers[ptr] = self._new_register(
                    parts=[ptr],
                    state=qt.basis(ptr.dim, 0, dtype=qt.data.CSR),
                    vm=vm,
                )

            for ptr in remaining_subsystem:
                vm.registers[ptr] = reg.sort()

            _targets = [t for t in _targets if t not in system]

        vm.stack.push(AnalogVMNULL)
