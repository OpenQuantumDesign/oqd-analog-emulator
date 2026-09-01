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

from oqd_analog_emulator.method_table import QubitName, QubitRegister
from oqd_analog_emulator.qutip_backend import QutipBackend


def interpreter(program):
    backend = QutipBackend()
    program, interp, output = backend.run(program)

    return output


class TestQutipBackend:
    @pytest.mark.parametrize(
        ("program", "expected"),
        [
            ("1", 1),
            ("1+2", 3),
            ("1-2", -1),
            ("1*-1", -1),
            ("1*2", 2),
            ("1/2", 0.5),
            ("2^3", 8),
            ("1+2*3-(5/5)^6", 6),
            ("imag(0)", 0),
            ("imag(1j)", 1),
            ("real(1j)", 0),
            ("real(-1)", -1),
            ("cos(0)", 1),
            ("heaviside(-2)", 0),
            ("heaviside(cos(0))", 1),
            ("atan2(0, 1)", 0),
            ("atan2(1, 0)", np.pi / 2),
            ("atan2(1, -1)", 3 * np.pi / 4),
            ("atan2(-1, -1)", -3 * np.pi / 4),
            ("a = 1 \n a", 1),
            ("a = 2 \n b = 3 \n c = a * b \n c", 6),
            ("a = 2 \n b = 3 \n c = a + b \n c", 5),
            ("a = 2 \n b = 3 \n c = a - b \n c", -1),
            ("a = 6 \n b = 3 \n c = a / b \n c", 2),
            ("a = 3 \n b = 2 \n c = a ^ b \n c", 9),
        ],
    )
    def test_qutip_math(self, program, expected):
        output = interpreter(program)
        assert output == expected

    @pytest.mark.parametrize(
        "program, expected",
        [
            ("3+5", 8),
            ("3.02+5.01", 8.03),
            ("3-5", -2),
            ("-3.02+5.01", 1.99),
            ("3*5", 15),
            ("15/2", 7.5),
            ("3^2.01", 9.10),
            ("sin(0.25)", 0.2474),
            ("tan(0.205)", 0.208),
            ("2*3 + 5*(1j)", 6 + 5j),
            ("1+2*3 + 9 - 0.1 + 7*(2+3*5+(10/3))", 158.233),
            ("sin(exp(2))", 0.894),
        ],
    )
    def test_qutip_math_approx(self, program, expected):
        output = interpreter(program)
        assert pytest.approx(output, 0.001) == expected

    @pytest.mark.parametrize(
        ("program", "expected"),
        [
            ("true", 1),
            ("false", 0),
            ("not true", 0),
            ("not false", 1),
            ("true and false", 0),
            ("true or false", 1),
            ("true && false", 0),
            ("true || false", 1),
            ("true and true", 1),
            ("false and false", 0),
            ("false or false", 0),
            ("a = true \n b = false \n a or b", 1),
            ("a = true \n b = false \n a and b", 0),
            ("a = true \n b = false \n a || b", 1),
            ("a = true \n b = false \n a && b", 0),
            ("a = true \n not a", 0),
            ("b = false \n not b", 1),
            ("a = true \n !a", 0),
            ("b = false \n !b", 1),
        ],
    )
    def test_qutip_bool(self, program, expected):
        output = interpreter(program)
        assert output == expected

    @pytest.mark.parametrize(
        ("program", "expected"),
        [
            ("1 == 2", 0),
            ("1 >= 2", 0),
            ("1 <= 2", 1),
            ("1 != 2", 1),
            ("1 > 2", 0),
            ("1 < 2", 1),
            ("a = true \n b = false \n a == b", 0),
            ("a = true \n b = true \n a == b", 1),
            ("a = true \n b = false \n a != b", 1),
            ("a = true \n b = true \n a != b", 0),
            ("a = 1 \n b = 2 \n a == b", 0),
            ("a = 1 \n b = 1 \n a == b", 1),
            ("a = 1 \n b = 2 \n a >= b", 0),
            ("a = 1 \n b = 2 \n a <= b", 1),
            ("a = 1 \n b = 2 \n a > b", 0),
            ("a = 1 \n b = 2 \n a < b", 1),
        ],
    )
    def test_qutip_comparators(self, program, expected):
        output = interpreter(program)
        assert output == expected

    @pytest.mark.parametrize(
        ("program", "expected"),
        [
            ("[1, 2, 3]", [1, 2, 3]),
            ("[1]", [1]),
            ("[]", []),
            # ("a = [1, 2, 3] \n [a[0], a[1]]", [1, 2]),
        ],
    )
    def test_qutip_list(self, program, expected):
        output = interpreter(program)
        assert all(output) == all(expected)

    @pytest.mark.parametrize(
        ("program", "expected"),
        [
            ("[[1], 2, 3]", [[1], 2, 3]),
            ("[[1, 2], 3]", [[1, 2], 3]),
            ("[[2], [3], [5, [6]]]", [[2], [3], [5, [6]]]),
        ],
    )
    def test_qutip_nested_list(self, program, expected):
        output = interpreter(program)
        assert all(output) == all(expected)

    @pytest.mark.parametrize(
        ("program", "expected"),
        [
            ("if (true) {\n a = 5} \n a", 5),
            ("a = 1 \n if (false) {\n a = 5} \n a", 1),
            ("a = 1 \n if (a > 0) {\n a = 5} \n a", 5),
            ("a = 1 \n if (a < 1) {\n a = 5} \n a", 1),
        ],
    )
    def test_qutip_if(self, program, expected):
        output = interpreter(program)
        assert output == expected

    @pytest.mark.parametrize(
        ("program", "expected"),
        [
            ("a = 1 \n if (true) {\n a = 5} \n else {\n a = 2} \n a", 5),
            ("a = 1 \n if (false) {\n a = 5} \n else {\n a = 2} \n a", 2),
            ("a = 1 \n if (a > 0) {\n a = 5} \n else {\n a = 2} \n a", 5),
            ("a = 1 \n if (a < 1) {\n a = 5} \n else {\n a = 2} \n a", 2),
        ],
    )
    def test_qutip_if_else(self, program, expected):
        output = interpreter(program)
        assert output == expected

    @pytest.mark.parametrize(
        ("program", "expected"),
        [
            ("x = 2 \n while (x > 0) {\n x = x - 1} \n x", 0),
            ("x = 2 \n while (true) {\n x = x + 1 \n if (x == 3) {\n break}} \n x", 3),
            ("x = 2 \n while (false) {\n x = x + 1 \n if (x == 3) {\n break}} \n x", 2),
            (
                "x = 2 \n a = 1 \n while (true) {\n x = x + 1 \n if (x == 5) {\n break} \n if (a>1) {\n continue}\n a = a + 1 \n} \n a",
                2,
            ),
            (
                "x = 2 \n a = 1 \n while (true) {\n x = x + 1 \n if (x == 5) {\n break} \n if (a>1) {\n continue}\n a = a + 1 \n} \n x",
                5,
            ),
        ],
    )
    def test_qutip_while(self, program, expected):
        output = interpreter(program)
        assert output == expected

    @pytest.mark.parametrize(
        ("program", "expected"),
        [
            ("r = qreg(2)", []),
            ("r = qreg(2) \n initialize(r)", []),
            ("r = qreg(2) \n initialize(r[0])", []),
            ("r = qreg(2) \n initialize([r[0], r[1]])", []),
            ("r = qreg(2) \n q0 = r[0]", []),
            ("r = qreg(2) \n q0 = r[0] \n q1 = r[1]", []),
        ],
    )
    def test_qutip_qubit_registers(self, program, expected):
        output = interpreter(program)
        assert output == expected

    @pytest.mark.parametrize(
        ("program", "expected"),
        [
            ("a = 1 \n b = a \n b", 1),
            ("a = 1 \n b = a \n c = b \n c", 1),
            ("a = 1 \n b = a \n c = b + 1 \n c", 2),
            ("a = 0 \n b = a \n c = sin(b) \n c", np.sin(0)),
            ("a = 0 \n c = a \n b = 1 \n c = b \n c", 1),
            (
                "r = qreg(2) \n q = r \n initialize(q) \n evolve(%X %@ %X, 1, q) \n result = evolve(%X, 1, q[0])",
                [],
            ),
        ],
    )
    def test_qutip_alias(self, program, expected):
        output = interpreter(program)
        assert output == expected

    @pytest.mark.parametrize(
        ("program", "expected"),
        [
            ("a = [1, 2] \n a[0]", 1),
            ("a = [1, 2] \n a[1]", 2),
            (
                "r = qreg(2) \n r[0]",
                QubitRegister(
                    name=[QubitName(name="r", index=0, dim=2)],
                    time_last_updated=0,
                    state=[],
                ),
            ),
            (
                "r = qreg(2) \n r[1]",
                QubitRegister(
                    name=[QubitName(name="r", index=1, dim=2)],
                    time_last_updated=0,
                    state=[],
                ),
            ),
        ],
    )
    def test_qutip_extract(self, program, expected):
        output = interpreter(program)
        assert output == expected

    @pytest.mark.parametrize(
        ("program", "expected"),
        [
            ("a = [1, 2] \n b = a \n b[0]", 1),
            ("a = [1, 2] \n b = a \n b[1]", 2),
            (
                "r = qreg(2) \n q = r \n q[0]",
                QubitRegister(
                    name=[QubitName(name="r", index=0, dim=2)],
                    time_last_updated=0,
                    state=[],
                ),
            ),
            (
                "r = qreg(2) \n q = r \n q[1]",
                QubitRegister(
                    name=[QubitName(name="r", index=1, dim=2)],
                    time_last_updated=0,
                    state=[],
                ),
            ),
        ],
    )
    def test_qutip_alias_extract(self, program, expected):
        output = interpreter(program)
        assert output == expected

    @pytest.mark.xfail(raises=ValueError, reason="Out-of-bounds indexing of array")
    @pytest.mark.parametrize(
        "program",
        [
            "a = [1, 2] \n a[2]",
            "r = qreg(2) \n r[2]",
        ],
    )
    def test_xfail_qutip_extract(self, program):
        interpreter(program)
