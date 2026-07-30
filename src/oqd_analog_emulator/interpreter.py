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
    qubits: list[tuple[str, int]]
    time: int
    state: object
    
    @property
    def n(self):
        return len(self.qubits)


class ModeObject:
    name: tuple[str, int]
    time: int
    state: object

class MethodTable:
    def __init__(self, n_shots = 10, fock_cutoff = 4, dt = 0.1):
        self._n_shots = n_shots
        self._fock_cutoff = fock_cutoff
        self._dt = dt
        self.stack = []
        self.store = {}
        self.GLOBAL_T = 0.0
    
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
    
    def run_GLOBALI(self, name):
        if name not in self.store.keys():
            self.store[name] = None

    def run_CONSTI(self, value):
        self.push(value)
    
    def run_STORE(self, name):
        self.store[name] = self.pop()

    def run_LOAD(self, name):
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

    def run_ADDI(self):
        self.push(self.pop() + self.pop())
    
    def run_SUBI(self):
        self.push(-self.pop() + self.pop())

    def run_MULI(self):
        self.push(self.pop() * self.pop())
    
    def run_DIVI(self):
        denom = self.pop()
        num = self.pop()
        self.push(num / denom)
    
    def run_POWI(self):
        exponent = self.pop()
        base = self.pop()
        self.push(base ** exponent)
    
    def run_KRONI(self):
        op2 = self.pop()
        op1 = self.pop()
        self.push(qt.tensor(op1, op2))
    
    def run_IMAGI(self):
        self.push(1j)
    
    def run_NOTI(self):
        self.push(not self.pop())
    
    def run_ANDI(self):
        self.push(self.pop() and self.pop())
    
    def run_ORI(self):
        self.push(self.pop() or self.pop())
    
    def run_EQI(self):
        self.push(self.pop() == self.pop())
    
    def run_NEQI(self):
        self.push(self.pop() != self.pop())
    
    def run_LTI(self):
        rhs = self.pop()
        lhs = self.pop()
        self.push(lhs < rhs)
   
    def run_LTEQI(self):
        rhs = self.pop()
        lhs = self.pop()
        self.push(lhs <= rhs)
    
    def run_GTI(self):
        rhs = self.pop()
        lhs = self.pop()
        self.push(lhs > rhs)   

    def run_GTEQI(self):
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
        self.push(self.store[name][index])
    
    def run_QREG(self, name, size):
        self.store[name] = []
        for n in range(size):
            obj = QubitObject()
            obj.register = (name, n)
            obj.state = []
            obj.time = self.GLOBAL_T
            self.store[name] += [obj]
    
    def run_MREG(self, name, size):
        self.store[name] = []
        for n in range(size):
            obj = ModeObject()
            self.store[name] += [obj]
        
    
    def _evolve(self, hamiltonian, duration, targets):
        
        tspan = np.linspace(0, duration, round(duration / 0.1)).tolist()
        results = {}

        for target in targets:
            start_runtime = time.time()
            result_qobj = qt.sesolve(
                hamiltonian,
                target.state, # Tensor product
                tspan,
                options={"store_states": True},
            )
            # print(self.results.runtime)
            target.time = time.time() - start_runtime + target.time
            # self.results.times.extend([t + self.results.times[-1] for t in tspan][1:])

            # for idx, key in enumerate(self.results.metrics.keys()):
            #     self.results.metrics[key].extend(result_qobj.expect[idx].tolist()[1:])
                
            target.state = result_qobj.final_state
            results[target] = result_qobj.final_state.full().squeeze()
            # self.current_state = result_qobj.final_state
        
        self.GLOBAL_T += duration
        self.push(results)
    
    def _measure(self, targets):
        counts = {}
        for target in targets:
            probs = np.power(np.abs(target.state.full()), 2).squeeze()
            n_shots = 10
            inds = np.random.choice(len(probs), size=n_shots, p=probs)
            # print(inds)
            opts = len(targets) * [[0,1]]
            bases = list(itertools.product(*opts))
            # print(bases)
            shots = np.array([bases[ind] for ind in inds])
            bitstrings = ["".join(map(str, shot)) for shot in shots]
            counts[target] = {
                bitstring: bitstrings.count(bitstring) for bitstring in bitstrings
            }
        
        self.push(counts)

    
    def _initialize(self, targets):
        dims = len(targets) * [2]
        # print(dims)

        # self.results.times.append(0.0)
        for target in targets:
            target.state = qt.tensor([qt.basis(d, 0) for d in dims])
            target.time = self.GLOBAL_T


    
class Interpreter:
    def __init__(self, graph: ControlFlowGraph, n_shots: int = 10, fock_cutoff: int = 4, dt: float = 0.1):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.method_table = MethodTable(n_shots, fock_cutoff, dt)
        
    def get_block(self, node: int = 0):
        return self.graph.blocks[node]

    def evaluate(self, stmt):
        instructions = QutipBackendInstructions()(stmt)
        # print(instructions)
        self.method_table.run(instructions)
    
    def run(self):
        node = 1
        current_block = self.get_block(node)
        
        while(current_block.kind != "stop"):
            stmt = current_block.stmt
            
            if (current_block.kind == "branch"):
                self.evaluate(stmt)
                cond = self.method_table.pop()
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
        return self.method_table.get_store()
    
    # def results(self):
    #     return self.method_table.get_results()
        
    
