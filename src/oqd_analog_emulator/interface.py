# Copyright 2024 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Dict, Literal, Tuple, Union

import qutip as qt

from pydantic import ConfigDict
from pydantic.types import NonNegativeInt

from oqd_compiler_infrastructure import VisitableBaseModel

########################################################################################

from oqd_core.interface.math import MathExpr
from oqd_core.backend.metric import *

########################################################################################

__all__ = ["QutipOperation", "QutipExperiment", "TaskArgsQutip", "QutipExpectation"]


class QutipExpectation(VisitableBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    operator: List[Tuple[qt.Qobj, MathExpr]]


class TaskArgsQutip(VisitableBaseModel):
    """
    Class representing args for QuTip

    Attributes:
        layer (str): the layer of the experiment (analog, atomic)
        n_shots (Union[int, None]): number of shots requested
        fock_cutof (int): fock_cutoff for QuTip simulation
        dt (float): timesteps for discrete time
        metrics (dict): metrics which should be computed for the experiment. This does not require any Measure instruction in the analog layer.
    """

    layer: Literal["analog"] = "analog"
    n_shots: Union[int, None] = 10
    fock_cutoff: int = 4
    dt: float = 0.1
    metrics: Dict[
        str, Union[EntanglementEntropyReyni, EntanglementEntropyVN, QutipExpectation]
    ] = {}


class QutipOperation(VisitableBaseModel):
    """
    Class representing a quantum operation in QuTip

    Attributes:
        hamiltonian (List[qt.Qobj, str]): Hamiltonian to evolve by
        duration (float): Duration of the evolution
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    hamiltonian: List[Tuple[qt.Qobj, MathExpr]]
    duration: float


class QutipMeasurement(VisitableBaseModel):
    pass


class QutipExperiment(VisitableBaseModel):
    """
    Class representing a quantum experiment in qutip

    Attributes:
        instructions (List[QutipOperation]): List of quantum operations to apply
        n_qreg (NonNegativeInt): Number of qubit quantum registers
        n_qmode (NonNegativeInt): Number of modal quantum registers
        args (TaskArgsQutip): Arguments for the experiment
    """

    instructions: list[Union[QutipOperation, QutipMeasurement]]
    n_qreg: NonNegativeInt
    n_qmode: NonNegativeInt
