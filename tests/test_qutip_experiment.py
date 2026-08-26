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
import pytest
from oqd_core.interface.analog.expr import (
    Annihilation,
    Creation,
    Identity,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
)

from oqd_analog_emulator.qutip_backend import QutipBackend


def get_amplitude_arrays(state: list):
    real_amplitudes, imag_amplitudes = [], []
    for x in state:
        real_amplitudes.append(x.real)
        imag_amplitudes.append(x.imag)
    return real_amplitudes, imag_amplitudes


def assert_lists_close(list1, list2, tolerance=0.001):
    assert len(list1) == len(list2), "The input lists have different length"
    for i, (elem1, elem2) in enumerate(zip(list1, list2)):
        assert (
            abs(elem1 - elem2) <= tolerance
        ), "List elements {i}, {elem1} and {elem2}, are out of tolerance"



def interpreter(program):
    backend = QutipBackend()
    program, _, output = backend.run(program)
    return output


class TestQutipExperiment:
    def test_one_qubit_rabi_flopping(self):
        """One qubit rabi flopping"""
        program = """ 
            r = qreg(1) \n initialize(r) \n H = -(3.14159 / 4) %* %X \n
            evolve(H, 1, r) \n evolve(H, 1, r) \n evolve(H, 1, r) \n r
        """
        
        output = interpreter(program)
        qubit_obj = output[0][1]
        state = qubit_obj.state
        
        real_amplitudes, imag_amplitudes = get_amplitude_arrays(state)
        assert_lists_close(real_amplitudes, [-0.707, 0])
        assert_lists_close(imag_amplitudes, [0, 0.707])
    
    def test_bell_state_standard(self):
        """Standard Bell State preparation"""
        program = """
            pi = 3.14159 \n r = qreg(2) \n initialize(r) \n evolve(%I %@ %I, (3 * pi) / 2, r) \n 
            evolve(%X %@ %I, pi / 2, r) \n evolve(-1 %* (%Y %@ %I), pi / 4, r) \n 
            evolve(%Y %@ %I, pi / 4, r) \n evolve(%X %@ %X, pi / 4, r) \n 
            evolve(-1 %* (%I %@ %X), pi / 4, r) \n evolve(-1 %* (%X %@ %I), pi / 4, r) \n 
            evolve(-1 %* (%Y %@ %I), pi / 4, r) \n evolve(%I %@ %I, pi / 4, r) \n r
        """
        
        output = interpreter(program)
        qubit_obj = output[0][1]
        state = qubit_obj.state
        
        real_amplitudes, imag_amplitudes = get_amplitude_arrays(state)
        assert_lists_close(real_amplitudes, [0.707, 0, 0, 0.707])
        assert_lists_close(imag_amplitudes, [0, 0, 0, 0])

    def test_ghz_state(self):
        """Standard GHz State preparation"""
        program = """ 
            pi = 3.14159 \n r = qreg(3) \n initialize(r) \n evolve(%I %@ %I %@ %I, (3 * pi) / 2, r) \n
            evolve(%X %@ %I %@ %I, pi / 2, r) \n evolve(-1 %* (%Y %@ %I %@ %I), pi / 4, r) \n 
            evolve(%Y %@ %I %@ %I, pi / 4, r) \n evolve(%X %@ %X %@ %I, pi / 4, r) \n 
            evolve(-1 %* (%I %@ %X %@ %I), pi / 4, r) \n evolve(-1 %* (%X %@ %I %@ %I), pi / 4, r) \n
            evolve(-1 %* (%Y %@ %I %@ %I), pi / 4, r) \n evolve(%I %@ %I %@ %I, pi / 4, r) \n 
            evolve(%Y %@ %I %@ %I, pi / 4, r) \n evolve(%X %@ %I %@ %X, pi / 4, r) \n 
            evolve(-1 %* (%I %@ %I %@ %X), pi / 4, r) \n evolve(-1 %* (%X %@ %I %@ %I), pi / 4, r) \n
            evolve(-1 %* (%Y %@ %I %@ %I), pi / 4, r) \n evolve(%I %@ %I %@ %I, pi / 4, r) \n r
        """
        
        output = interpreter(program)
        qubit_obj = output[0][1]
        state = qubit_obj.state
        
        real_amplitudes, imag_amplitudes = get_amplitude_arrays(state)
        assert_lists_close(real_amplitudes, [0.707, 0, 0, 0, 0, 0, 0, 0.707])
        assert_lists_close(imag_amplitudes, [0, 0, 0, 0, 0, 0, 0, 0])
    
    def test_identity_operation_simple(self):
        """Simple Identity operation using inverse"""
        program = """ 
            pi = 3.14159 \n r = qreg(1) \n initialize(r) \n evolve(-1 %* %X, 1, r) \n
            evolve(%X, 1, r) \n r
        """
        
        output = interpreter(program)
        qubit_obj = output[0][1]
        state = qubit_obj.state
        
        real_amplitudes, imag_amplitudes = get_amplitude_arrays(state)
        assert_lists_close(real_amplitudes, [1, 0])
        assert_lists_close(imag_amplitudes, [0, 0])
    
    def test_identity_operation_nested(self):
        """Nested Identity operation using inverse"""
        program = """ 
            pi = 3.14159 \n r = qreg(1) \n initialize(r) \n evolve(-1 %* %X, 1, r) \n
            evolve(%X, 1, r) \n evolve(-1 %* %X, 1, r) \n evolve(%X, 1, r) \n 
            \n evolve(-1 %* %X, 1, r) \n evolve(%X, 1, r) \n r
        """
        
        output = interpreter(program)
        qubit_obj = output[0][1]
        state = qubit_obj.state
        
        real_amplitudes, imag_amplitudes = get_amplitude_arrays(state)
        assert_lists_close(real_amplitudes, [1, 0])
        assert_lists_close(imag_amplitudes, [0, 0])
    
    def test_identity_operation_three_qubit_simple(self):
        """Simple Identity operation using inverse for 3 qubits"""
        program = """ 
            pi = 3.14159 \n r = qreg(3) \n initialize(r) \n evolve(-1 %* (%X %@ %Y %@ %Z), 1, r) \n
            evolve(%X %@ %Y %@ %Z, 1, r) \n r
        """
        
        output = interpreter(program)
        qubit_obj = output[0][1]
        state = qubit_obj.state
        
        real_amplitudes, imag_amplitudes = get_amplitude_arrays(state)
        assert_lists_close(real_amplitudes, [1, 0, 0, 0, 0, 0, 0, 0])
        assert_lists_close(imag_amplitudes, [0, 0, 0, 0, 0, 0, 0, 0])
        
    def test_identity_operation_three_qubit_nested(self):
        """Nested Identity operation using inverse for 3 qubits"""
        program = """ 
            pi = 3.14159 \n r = qreg(3) \n initialize(r) \n evolve(-1 %* (%X %@ %Y %@ %Z), 1, r) \n
            evolve(%X %@ %Y %@ %Z, 1, r) \n evolve(-1 %* (%X %@ %X %@ %X), 1, r) \n 
            evolve(%X %@ %X %@ %X, 1, r) \n evolve(-1 %* (%I %@ %X %@ %I), 1, r) \n
            evolve(%I %@ %X %@ %I, 1, r) \n r
        """
        
        output = interpreter(program)
        qubit_obj = output[0][1]
        state = qubit_obj.state
        
        real_amplitudes, imag_amplitudes = get_amplitude_arrays(state)
        assert_lists_close(real_amplitudes, [1, 0, 0, 0, 0, 0, 0, 0])
        assert_lists_close(imag_amplitudes, [0, 0, 0, 0, 0, 0, 0, 0])
    
    def test_one_qubit_rabi_flopping_canonicalization(self):
        """One qubit rabi flopping canonicalization"""
    
        program = """ 
            r = qreg(1) \n initialize(r) \n Hx = -1 %* ((3.14159 / 8) %* (2 %* %X)) \n 
            evolve(Hx, 1, r) \n evolve(Hx, 1, r) \n evolve(Hx, 1, r) \n r
        """
        output = interpreter(program)
        qubit_obj = output[0][1]
        state = qubit_obj.state
        
        real_amplitudes, imag_amplitudes = get_amplitude_arrays(state)
        assert_lists_close(real_amplitudes, [-0.707, 0])
        assert_lists_close(imag_amplitudes, [0, 0.707])
       
    # def test_bell_state_canonicalization(self):
    #     """Standard Bell State preparation canonicalization"""
    #     program = """
    #         pi = 3.14159 \n r = qreg(2) \n initialize(r) \n evolve(%I %@ %I, (3 * pi) / 2, r) \n 
    #         evolve(%X %@ %I, pi / 2, r) \n evolve((-1 * 0.5) %* (%Y %@ (2 %* %I)), pi / 4, r) \n 
    #         evolve(%Y %@ %I, pi / 4, r) \n evolve(%X %@ (%I %* %X %* %I), pi / 4, r) \n 
    #         evolve(-1 %* (%I %@ %X), pi / 4, r) \n evolve(-1 %* (%X %@ %I), pi / 4, r) \n 
    #         evolve((-1 * 0.5) %* (%Y %@ (2 %* %I)), pi / 4, r) \n evolve(%I %@ %I, pi / 4, r) \n r
    #     """
        
    #     output = interpreter(program)
    #     qubit_obj = output[0][1]
    #     state = qubit_obj.state
        
    #     real_amplitudes, imag_amplitudes = get_amplitude_arrays(state)
    #     assert_lists_close(real_amplitudes, [0.707, 0, 0, 0.707])
    #     assert_lists_close(imag_amplitudes, [0, 0, 0, 0])