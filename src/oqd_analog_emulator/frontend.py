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
from collections.abc import Callable

import numpy as np
import qutip as qt
import typer
from oqd_compiler_infrastructure import CFGBlockAccumulator, RelabelCFGBlocks
from oqd_core.analysis.analog import AnalogCFGBuilder
from oqd_core.analysis.analog.dim_checker import DimensionChecker, DInvalid
from oqd_core.analysis.analog.reaching_def import AvailableVariableAnalysis
from oqd_core.analysis.analog.type_checker import AnalogTypeChecker
from oqd_core.analysis.analog.types import (
    TBool,
    TComplex,
    TFloat,
    TInt,
    TList,
    TOp,
    TQRegElem,
)
from oqd_core.frontend.analog import parse_analog

from oqd_analog_emulator.instructions import ListTerminators
from oqd_analog_emulator.interpreter import AnalogInterpreter
from oqd_analog_emulator.method_table import QuantumRegisterPointer
from oqd_analog_emulator.qutip import QutipMethodTable as QutipMethodTable

########################################################################################


class AnalogREPR:
    STARTSTRING = f"""
{"=" * 80}
{"Welcome to the Analog Interpreter REPR for the Analog langugage of OQD's stack!":^80}
{"=" * 80}
    """.strip()

    def __init__(self, *, method_table, options, **kwargs):
        self.interp = AnalogInterpreter(
            method_table=method_table, options=options, **kwargs
        )

        self.avail_checker = AvailableVariableAnalysis()
        self.type_checker = AnalogTypeChecker()
        self.dim_checker = DimensionChecker()

    def _infer_type(self, value):
        match value:
            case QuantumRegisterPointer():
                return TQRegElem
            case bool():
                return TBool
            case int() | np.int64():
                return TInt
            case float():
                return TFloat
            case complex():
                return TComplex
            case qt.Qobj() | qt.QobjEvo():
                return TOp
            case list():
                return TList[self._infer_type(value[1])]
            case Callable():
                self._infer_type(value(0, 0))

        raise TypeError()

    def _infer_dim(self, value):
        match value:
            case QuantumRegisterPointer():
                return [value.dim]
            case qt.Qobj() | qt.QobjEvo():
                return value.dims[0]
            case Callable():
                return self._infer_dim(value(0, 0))
            case list():
                elem_dim = self._infer_dim(value[1])
                return [
                    elem_dim[0] if isinstance(e, QuantumRegisterPointer) else elem_dim
                    for e in value
                    if not isinstance(e, ListTerminators)
                ]
            case _:
                return DInvalid

    def _get_envs(self):
        avail_env = {x for x in self.interp.vm.store.keys() if x[0] not in ["$", "#"]}

        type_env = {
            k: self._infer_type(v)
            for k, v in self.interp.vm.store.items()
            if k[0] not in ["$", "#"]
        }

        dim_env = {
            k: self._infer_dim(v)
            for k, v in self.interp.vm.store.items()
            if k[0] not in ["$", "#"]
        }

        return avail_env, type_env, dim_env

    def compile(self, program: str):
        circuit = parse_analog(program)
        cfg = AnalogCFGBuilder()(circuit)
        cfg = CFGBlockAccumulator()(cfg)
        cfg = RelabelCFGBlocks()(cfg)

        avail_env, type_env, dim_env = self._get_envs()

        self.avail_checker.analyze(
            cfg,
            initial_state={
                node: avail_env if node == 0 else self.avail_checker.lattice.bottom()
                for node in cfg.nodes()
            }
            if avail_env
            else None,
        )
        self.type_checker.analyze(
            cfg,
            initial_state={
                node: type_env if node == 0 else self.type_checker.lattice.top()
                for node in cfg.nodes()
            }
            if type_env
            else None,
        )
        self.dim_checker.analyze(
            cfg,
            initial_state={
                node: dim_env if node == 0 else self.dim_checker.lattice.top()
                for node in cfg.nodes()
            }
            if dim_env
            else None,
        )

        return cfg

    def _run_block(self, block, *, avail_env=None, type_env=None, dim_env=None):
        success = False
        try:
            cfg = self.compile(block)

        except Exception as e:
            print(f"{e.__class__.__name__}: {e}", flush=True)
            return success

        try:
            res = self.interp.run(cfg=cfg)
            success = True
            print(f" : {res}")
        except Exception as e:
            print(f"{e.__class__.__name__}: {e}", flush=True)

        return success

    def run(self, program: str = ""):
        print(AnalogREPR.STARTSTRING)

        ANSIGREEN = "\001\033[1;32m\002"
        ANSIRED = "\001\033[1;31m\002"
        ANSIRESET = "\001\033[0m\002"

        success = True
        if program:
            for n, line in enumerate(program.splitlines()):
                print(f"{ANSIGREEN}{'>>' if n == 0 else ' >'}{ANSIRESET} {line}")

            success = self._run_block(program)

        while True:
            ansi_color = ANSIGREEN if success else ANSIRED

            lines = []
            firstline = input(f"{ansi_color}>>{ANSIRESET} ")
            lines.append(firstline)
            while True:
                if lines[-1] in ["", "exit", "exit()", "reset", "reset()"]:
                    break

                lines.append(input(f" {ansi_color}>{ANSIRESET} "))

            match lines[-1].strip():
                case "exit" | "exit()":
                    break
                case "reset" | "reset()":
                    self.interp.reset()
                    continue

            block = "\n".join(lines)
            success = self._run_block(block)


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
