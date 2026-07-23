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

from typing import List

from oqd_core.analysis.utils import ControlFlowGraph
from oqd_analog_emulator.rewrite import QutipBackendInstructions


class QubitObject():
    register: tuple[str, int]
    t: int
    state: List[int]


class QubitRegister():
    qubits: List[tuple[str, int]]
    t: int
    state: List[int]
    n: int # len(state)


code = [
   ('GLOBALI', 'x'),
   ('CONSTI', 4),
   ('STORE', 'x'),
   ('GLOBALI', 'y'),
   ('CONSTI', 5),
   ('STORE', 'y'),
   ('GLOBALI', 'd'),
   ('LOAD', 'x'),
   ('LOAD', 'x'),
   ('MULI',),
   ('LOAD', 'y'),
   ('LOAD', 'y'),
   ('MULI',),
   ('ADDI',),
   ('STORE', 'd'),
   ('LOAD', 'd'),
]

class Evaluator():
    def __init__(self):
        self.stack = []
        self.register_stack = []
        self.store = {}
    
    def get_store(self):
        return self.store
        
    def push(self, item):
        self.stack.append(item)
    
    def pop(self):
        return self.stack.pop()
    
    def run(self, code):
        self.pc = 0
        while self.pc < len(code):
            op, *opargs = code[self.pc]
            getattr(self, f'run_{op}')(*opargs)
            self.pc += 1
    
    def run_GLOBALI(self, name):
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
    

class Interpreter():
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
        return self.evaluator.get_store()
    
    def run(self):
        node = 0
        current_block = self.get_block(node)
        
        while(current_block.kind != "stop"):
            
            if current_block.kind == "stmt":
                stmt = current_block.stmt
                if stmt:
                    # print(stmt)
                    store = self.evaluate(stmt)
                    print(store)
            node += 1
            current_block = self.get_block(node)
            
            

        
    
    