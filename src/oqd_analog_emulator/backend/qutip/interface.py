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

from typing import Annotated, Union

import qutip as qt

from pydantic import ConfigDict, Discriminator
from pydantic.types import NonNegativeInt

from oqd_compiler_infrastructure import TypeReflectBaseModel


########################################################################################

__all__ = ["QutipOperation", "QutipExperiment", "QutipExpectation"]


class QutipExpectation(TypeReflectBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    operator: qt.QobjEvo


class QutipOperation(TypeReflectBaseModel):
    """
    Class representing a quantum operation in QuTip

    Attributes:
        hamiltonian (List[qt.Qobj, str]): Hamiltonian to evolve by
        duration (float): Duration of the evolution
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    hamiltonian: qt.QobjEvo
    duration: float


class QutipMeasurement(TypeReflectBaseModel):
    pass


class QutipInitialization(TypeReflectBaseModel):
    pass


class QutipExperiment(TypeReflectBaseModel):
    """
    Class representing a quantum experiment in qutip

    Attributes:
        instructions (List[QutipOperation]): List of quantum operations to apply
        n_qreg (NonNegativeInt): Number of qubit quantum registers
        n_qmode (NonNegativeInt): Number of modal quantum registers
        args (TaskArgsQutip): Arguments for the experimentN
    """

    instructions: list[
        Annotated[
            Union[QutipOperation, QutipMeasurement, QutipInitialization],
            Discriminator(discriminator="class_"),
        ]
    ]
    n_qreg: NonNegativeInt
    n_qmode: NonNegativeInt
