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

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np

from oqd_dataschema import AnalogEmulatorDataGroup, Datastore, Dataset

if TYPE_CHECKING:
    from oqd_analog_emulator.conversion import QutipExperimentVM
    from oqd_analog_emulator.interface import TaskArgsQutip

########################################################################################

__all__ = [
    "EMULATION_GROUP_KEY",
    "METRIC_LABELS_ATTR",
    "vm_to_datastore",
]

EMULATION_GROUP_KEY = "emulation"
METRIC_LABELS_ATTR = "metric_labels"

########################################################################################


def vm_to_datastore(vm: QutipExperimentVM, args: TaskArgsQutip) -> Datastore:
    """
    Build a [`Datastore`][oqd_dataschema.datastore.Datastore] from VM run data.

    Expects ``vm`` to hold plain numpy-friendly collections populated during
    simulation (no ``TaskResultAnalog`` intermediate).
    """
    times = np.asarray(vm.times, dtype=np.float64)

    metrics_dataset = None
    if vm.metric_labels:
        metrics_data = np.column_stack(
            [
                np.asarray(vm.metrics[label], dtype=np.float64)
                for label in vm.metric_labels
            ]
        )
        metrics_dataset = Dataset(
            data=metrics_data,
            attrs={METRIC_LABELS_ATTR: ",".join(vm.metric_labels)},
        )

    state_dataset = None
    if vm.state_trajectory:
        state_dataset = Dataset(data=np.vstack(vm.state_trajectory))

    measurements_dataset = None
    if vm.measurements is not None and vm.measurements.size > 0:
        measurements_dataset = Dataset(
            data=vm.measurements,
            attrs={
                "axis_0": "shots",
                "axis_1": "qubits",
            },
        )

    try:
        pkg_version = version("oqd-analog-emulator")
    except Exception:
        pkg_version = "unknown"

    group_attrs = {
        "backend": "qutip",
        "version": pkg_version,
        "dt": args.dt,
        "fock_cutoff": args.fock_cutoff,
        "layer": args.layer,
        "runtime": vm.runtime,
    }
    if args.n_shots is not None:
        group_attrs["n_shots"] = args.n_shots

    emulation = AnalogEmulatorDataGroup(
        attrs=group_attrs,
        times=Dataset(data=times),
        metrics=metrics_dataset,
        state=state_dataset,
        measurements=measurements_dataset,
    )

    return Datastore(groups={EMULATION_GROUP_KEY: emulation})
