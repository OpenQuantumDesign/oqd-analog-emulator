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

import numpy as np
import time
import itertools
import qutip as qt
import math
from oqd_core.analysis.utils import ControlFlowGraph
from oqd_analog_emulator.instructions import QutipBackendInstructions


class QubitObject:
    name: tuple[str, int]
    time: int
    state: object
    

class QubitRegister:
    qubits: list[tuple[str, int]] = []
    time: int
    state: object
    
    @property
    def n(self):
        return len(self.qubits)


class ModeObject:
    name: tuple[str, int]
    time: int
    state: object

class Interpreter:
    def __init__(self, n_shots = 10, fock_cutoff = 4, dt = 0.1):
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
    
    def push(self, item):
        self.stack.append(item)
    
    def pop(self):
        if self.stack == []: return None
        return self.stack.pop()
    
    def peek(self):
        if self.stack:
            return self.stack[-1]
        return None
    
    def run(self, code):
        self.pc = 0
        while self.pc < len(code):
            op, *opargs = code[self.pc]
            getattr(self, f'run_{op}')(*opargs)
            self.pc += 1
    
    def run_GLOBAL(self, name):
        if name not in self.store.keys():
            self.store[name] = None

    def run_CONST(self, value):
        self.push(value)
    
    def run_STORE(self, name):
        self.store[name] = self.pop()

    def run_LOAD(self, name):
        if name not in self.store.keys():
            registers = self.registers[name]
            for name in registers:
                self.push(self.store[name])
        self.push(self.store[name])
    
    def run_FUNC(self, func):
        output = None
        operation = getattr(math, func, None)
        if operation is None:
            operation = getattr(np, func, None)
        if operation is None:
            raise ValueError("Unknown math function")
            
        match func:
            case "abs":
                output = abs(self.pop())
            case "heaviside":
                output = np.heaviside(self.pop(), 0)
            case "atan2":
                x = self.pop()
                y = self.pop()
                output = operation(y, x)
            case _:
                output = operation(self.pop())
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
        self.push(base ** exponent)
    
    def run_KRON(self):
        op2 = self.pop()
        op1 = self.pop()
        self.push(qt.tensor(op1, op2))
    
    def run_IMAG(self):
        self.push(1j)
    
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
    
    def run_EVOLVE(self):
        operands = []
        elem = self.pop()
        while elem:
            operands += [elem]
            elem = self.pop()
        
        targets = operands[:-2]
        duration = operands[-2]
        hamiltonian = operands[-1]
        self._evolve(hamiltonian, duration, targets)
        
    
    def run_INIT(self):
        targets = []
        elem = self.pop()
        while elem:
            targets += [elem]
            elem = self.pop()
        self._initialize(targets)
    
    def run_MEASURE(self):
        targets = []
        elem = self.pop()
        while elem:
            targets += [elem]
            elem = self.pop()
        self._measure(targets)
    
    def run_LIST(self, name):
        elem = self.pop()
        while elem:
            if self.store[name] is None:
                self.store[name] = [elem]
            else:
                self.store[name].insert(0, elem)
            elem = self.pop()
    
    def run_EXTRACT(self, name, index):
        self.push(self.store[(name, index)])
    
    def run_DEC_EX(self, name, extract, index):
        self.registers[name] = [(extract, index)]
    
    def run_QREG(self, name, size):
        self.registers[name] = []
        for n in range(size):
            obj = QubitObject()
            obj.name = (name, n)
            obj.state = []
            obj.time = self.GLOBAL_T
            self.store[(name, n)] = obj
            self.registers[name].append((name, n))
        
    
    def run_MREG(self, name, size):
        self.store[name] = []
        for n in range(size):
            obj = ModeObject()
            obj.name = (name, n)
            obj.state = []
            obj.time = self.GLOBAL_T
            self.store[(name, n)] = obj
        
    
    def _evolve(self, hamiltonian, duration, targets):
        
        tspan = np.linspace(0, duration, round(duration / self._dt)).tolist()
        results = {}
        states = []
        for target in targets:
            states += target.state
        print(states)
            
        start_runtime = time.time()
        result_qobj = qt.sesolve(
            hamiltonian,
            qt.tensor(qt.Qobj(states)), # Tensor product
            tspan,
            options={"store_states": True},
        )
        # print(self.results.runtime)
        target.time = time.time() - start_runtime + target.time
        self.GLOBAL_T += duration
        # self.results.times.extend([t + self.results.times[-1] for t in tspan][1:])

        # for idx, key in enumerate(self.results.metrics.keys()):
        #     self.results.metrics[key].extend(result_qobj.expect[idx].tolist()[1:])
            
        # target.state = result_qobj.final_state
        results = result_qobj.final_state.full().squeeze()
    
        register = QubitRegister()
        register.time = self.GLOBAL_T
        register.state = result_qobj.final_state
        for target in targets:
            if target.name not in register.qubits:
                register.qubits.append(target.name)
        
        for target in targets:
            self.store[target.name] = register
        
        self.push(results)
    
    def _measure(self, targets):
        counts = {}
        ind = 0
        for target in targets:
            probs = np.power(np.abs(target.state.full()), 2).squeeze()
            n_shots = self._n_shots
            inds = np.random.choice(len(probs), size=n_shots, p=probs)
            # print(inds)
            dims = 2
            if isinstance(target, ModeObject):
                dims = self._fock_cutoff
            opts = len(targets) * [list(range(dims))]
            bases = list(itertools.product(*opts))
            # print(bases)
            shots = np.array([bases[ind] for ind in inds])
            bitstrings = ["".join(map(str, shot)) for shot in shots]
            counts[ind] = {
                bitstring: bitstrings.count(bitstring) for bitstring in bitstrings
            }
            ind += 1
        
        self.push(counts)

    
    def _initialize(self, targets):
        # dims = len(targets) * [2]
        # print(dims)

        # self.results.times.append(0.0)
        for target in targets:
            if isinstance(target, QubitObject):
                target.state = qt.Qobj([1, 0])
            elif isinstance(target, ModeObject):
                target.state = qt.Qobj([self._fock_cutoff, 0])
            target.time = self.GLOBAL_T


    
class IRGenerator:
    def __init__(self, graph: ControlFlowGraph, n_shots: int = 10, fock_cutoff: int = 4, dt: float = 0.1):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.interpreter = Interpreter(n_shots, fock_cutoff, dt)
        self._fock_cutoff = fock_cutoff
        
    def get_block(self, node: int = 0):
        return self.graph.blocks[node]

    def evaluate(self, stmt):
        instructions = QutipBackendInstructions(fock_cutoff=self._fock_cutoff)(stmt)
        # print(instructions)
        self.interpreter.run(instructions)
    
    def run(self):
        node = 1
        current_block = self.get_block(node)
        
        while(current_block.kind != "stop"):
            stmt = current_block.stmt
            
            if (current_block.kind == "branch"):
                self.evaluate(stmt)
                cond = self.interpreter.pop()
                if cond:
                    node = next(key for key, val in current_block.edge_labels.items() if val == 'true')
                else:
                    node = next(key for key, val in current_block.edge_labels.items() if val == 'false')
                
            if current_block.kind == "stmt":
                if stmt:
                    # print(stmt)
                    self.evaluate(stmt)
                if current_block.succs:
                    current_block = current_block.succs[0]
                    continue
            # print(node)
            current_block = self.get_block(node)
            
    def status(self):
        return self.interpreter.get_store()
    
