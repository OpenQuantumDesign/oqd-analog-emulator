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


import pathlib
import readline as readline

import typer
from oqd_core.analysis.analog.cfg import AnalogCFGBuilder
from oqd_core.analysis.analog.symbol_table import AnalogSymbolTableBuilder
from oqd_core.analysis.analog.type_checker import AnalogTypeChecker
from oqd_core.compiler.analog.cfg_passes.walk import (
    canonicalize_math_cfg,
    canonicalize_operators_cfg,
)
from oqd_core.compiler.analog.verify.passes import (
    verify_hamiltonian_target_dim,
    verify_register_access_dim,
)
from oqd_core.frontend.analog import parse_analog

from oqd_analog_emulator.interpreter import AnalogInterpreter

########################################################################################


class AnalogREPR:
    def __init__(self, *, method_table, options, **kwargs):
        self.interp = AnalogInterpreter(
            method_table=method_table, options=options, **kwargs
        )

    def compile(self, program: str, *, type_check=True):
        # TODO: Enable forwarding of type checker and symbol table results
        # TODO: to following code to be executed

        circuit = parse_analog(program)
        cfg = AnalogCFGBuilder().run(circuit)

        canonicalize_operators_cfg(cfg)
        canonicalize_math_cfg(cfg)

        if type_check:
            checker = AnalogTypeChecker(cfg)

            symbol_analysis = AnalogSymbolTableBuilder(cfg, checker.dataflow_result)
            symbol_table = symbol_analysis.symbol_table

            verify_register_access_dim(cfg, symbol_table)
            verify_hamiltonian_target_dim(cfg, symbol_table)

        return cfg

    def _run_block(self, block, previous):
        success = False
        try:
            new_program = previous + "\n" + block
            # * Workaround invalild type checking by rerunning type check combining executed code and new code
            self.compile(new_program, type_check=True)

            cfg = self.compile(block, type_check=False)

        except Exception as e:
            print(f"{e.__class__.__name__}: {e}")

            new_program = previous
            cfg = None

        try:
            res = self.interp.run(cfg=cfg) if cfg else ""

            success = True

            print(f": {res}")
        except Exception as e:
            print(f"{e.__class__.__name__}: {e}")

        return new_program, success

    def run(self, program: str = ""):
        start_string = f"""
{"=" * 80}
{"Welcome to the Analog Interpreter REPR for the Analog langugage of OQD's stack!":^80}
{"=" * 80}
        
        """.strip()

        print(start_string)

        ANSIGREEN = "\001\033[1;32m\002"
        ANSIRED = "\001\033[1;31m\002"
        ANSIRESET = "\001\033[0m\002"

        success = True
        previous = ""
        if program:
            for n, line in enumerate(program.splitlines()):
                print(f"{ANSIGREEN}{'>>' if n == 0 else '>'}{ANSIRESET} {line}")

            previous, success = self._run_block(program, previous)

        while True:
            ansi_color = ANSIGREEN if success else ANSIRED

            lines = []
            firstline = input(f"{ansi_color}>>{ANSIRESET} ")
            lines.append(firstline)
            while True:
                if lines[-1] in ["", "exit", "exit()", "clear", "clear()"]:
                    break

                lines.append(input(f"{ansi_color}>{ANSIRESET} "))

            match lines[-1].strip():
                case "exit" | "exit()":
                    break
                case "clear" | "clear()":
                    self.interp.clear()
                    previous = ""
                    continue

            block = "\n".join(lines)

            previous, success = self._run_block(block, previous)


########################################################################################

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def run_analog_repr(
    program: str | None = typer.Option(
        None, "-c", "--code", help="Analog language code to run"
    ),
    program_file: pathlib.Path | None = typer.Option(
        None, "-s", "--s", help="Analog language script to run"
    ),
    method_table: str = typer.Option(
        "QutipMethodTable",
        "--mt",
        "--method-table",
        help="Analog language script to run",
    ),
    options: str | None = typer.Option(
        None, "--opt", "--options", help="Interpreter options"
    ),
    options_file: pathlib.Path | None = typer.Option(
        None, "--options-file", help="Interpreter options file"
    ),
):
    """
    Runs an Analog Interpreter REPR environment for the Analog language of OQD's stack.
    """

    if program:
        program = program.replace("\\n", "\n")

    analog_repr = AnalogREPR(method_table=method_table, options=options)

    analog_repr.run(program if program else "")
