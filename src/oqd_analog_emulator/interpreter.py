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
from oqd_core.analysis.utils import ControlFlowGraph
from oqd_core.backend.task import TaskResultAnalog
from oqd_analog_emulator.instructions import QutipBackendInstructions


class QubitObject:
    register: tuple[str, int]
    t: int
    state: list[int]


class QubitRegister:
    qubits: list[tuple[str, int]]
    t: int
    state: list[int]
    n: int # len(state)


class Evaluator:
    def __init__(self):
        self.stack = []
        self.store = {}
        self.GLOBAL_T = 0
        self.results = TaskResultAnalog(runtime=0)
    
    def get_store(self):
        return self.store
    
    def get_results(self):
        return self.results
        
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
        self.evolve(hamiltonian, duration, targets)
        
    
    def run_INIT(self):
        targets = []
        elem = self.pop()
        while elem:
            targets += [elem]
            elem = self.pop()
        self.initialize(targets)
    
    def run_MEASURE(self):
        targets = []
        elem = self.pop()
        while elem:
            targets += [elem]
            elem = self.pop()
        self.measure(targets)
    
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
            obj.t = self.GLOBAL_T
            self.store[name] += [obj]
    
    def evolve(self, hamiltonian, duration, targets):
        
        tspan = np.linspace(0, duration, round(duration / 0.1)).tolist()

        for target in targets:
            start_runtime = time.time()
            result_qobj = qt.sesolve(
                hamiltonian,
                self.current_state,
                tspan,
                options={"store_states": True},
            )
            self.results.runtime = time.time() - start_runtime + self.results.runtime
            self.results.times.extend([t + self.results.times[-1] for t in tspan][1:])

            # for idx, key in enumerate(self.results.metrics.keys()):
            #     self.results.metrics[key].extend(result_qobj.expect[idx].tolist()[1:])
                
            target.state = result_qobj.final_state

        self.results.state = list(
            result_qobj.final_state.full().squeeze(),
        )
    
    def measure(self, targets):
        for target in targets:
            probs = np.power(np.abs(self.current_state.full()), 2).squeeze()
            n_shots = 10
            inds = np.random.choice(len(probs), size=n_shots, p=probs)
            opts = self.current_state
            bases = list(itertools.product(*opts))
            shots = np.array([bases[ind] for ind in inds])
            bitstrings = ["".join(map(str, shot)) for shot in shots]
            self.results.counts = {
                bitstring: bitstrings.count(bitstring) for bitstring in bitstrings
            }

            self.results.state = list(
                self.current_state.full().squeeze(),
            )
    
    def initialize(self, targets):
        dims = len(targets) * [2]
        # print(dims)
        self.current_state = qt.tensor([qt.basis(d, 0) for d in dims])

        self.results.times.append(0.0)
        for target in targets:
            target.state = list(
                self.current_state.full().squeeze(),
            )
        
        self.results.state = list(
            self.current_state.full().squeeze(),
        )

    
class Interpreter:
    def __init__(self, graph: ControlFlowGraph):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.evaluator = Evaluator()
    
    def get_block(self, node: int = 0):
        return self.graph.blocks[node]

    def evaluate(self, stmt):
        instructions = QutipBackendInstructions()(stmt)
        # print(instructions)
        self.evaluator.run(instructions)
    
    def run(self):
        node = 1
        current_block = self.get_block(node)
        
        while(current_block.kind != "stop"):
            stmt = current_block.stmt
            
            if (current_block.kind == "branch"):
                self.evaluate(stmt)
                cond = self.evaluator.pop()
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
        return self.evaluator.get_store()
    
    def results(self):
        return self.evaluator.get_results()
        
    
