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

import json
from importlib.metadata import version
from typing import List, Optional, Union

import numpy as np

from oqd_core.backend.task import TaskArgsAnalog, TaskResultAnalog
from oqd_dataschema import AnalogEmulatorDataGroup, Datastore, Dataset

########################################################################################

__all__ = [
    "EMULATION_GROUP_KEY",
    "metric_labels_from_dataset",
    "task_result_to_datastore",
]

EMULATION_GROUP_KEY = "emulation"

########################################################################################


def _complex_list_to_array(values: List) -> np.ndarray:
    """Convert a list of ComplexFloat or complex values to a 1D complex128 array."""
    out = np.empty(len(values), dtype=np.complex128)
    for i, value in enumerate(values):
        if hasattr(value, "real") and hasattr(value, "imag"):
            out[i] = complex(value.real, value.imag)
        else:
            out[i] = complex(value)
    return out


def metric_labels_from_dataset(metrics_dataset: Dataset) -> List[str]:
    """Decode metric axis labels stored in a metrics dataset's attrs."""
    raw = metrics_dataset.attrs.get("metric_labels", "[]")
    return json.loads(raw)


def task_result_to_datastore(
    result: TaskResultAnalog,
    args: TaskArgsAnalog,
    *,
    state_trajectory: Optional[List[List]] = None,
    measurements: Optional[np.ndarray] = None,
    backend: str = "qutip",
) -> Datastore:
    """
    Build an [`Datastore`][oqd_dataschema.datastore.Datastore] from analog emulator output.

    Args:
        result: Raw simulation results from the QuTiP virtual machine.
        args: Task arguments used for the run (metadata source).
        state_trajectory: State vectors at each time step, if collected.
        measurements: Sampled qubit outcomes of shape ``(n_shots, n_qubits)``.
        backend: Backend identifier stored in group attrs.
    """
    times = np.asarray(result.times, dtype=np.float64)
    metric_labels = list(result.metrics.keys())

    metrics_dataset = None
    if metric_labels:
        metrics_data = np.column_stack(
            [np.asarray(result.metrics[label], dtype=np.float64) for label in metric_labels]
        )
        metrics_dataset = Dataset(
            data=metrics_data,
            attrs={"metric_labels": json.dumps(metric_labels)},
        )

    state_dataset = None
    if state_trajectory is not None and len(state_trajectory) > 0:
        state_rows = [_complex_list_to_array(row) for row in state_trajectory]
        state_dataset = Dataset(data=np.vstack(state_rows))

    measurements_dataset = None
    if measurements is not None and measurements.size > 0:
        measurements_dataset = Dataset(
            data=np.asarray(measurements, dtype=np.int64),
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
        "backend": backend,
        "version": pkg_version,
        "dt": args.dt,
        "fock_cutoff": args.fock_cutoff,
        "layer": args.layer,
    }
    if args.n_shots is not None:
        group_attrs["n_shots"] = args.n_shots
    if result.runtime is not None:
        group_attrs["runtime"] = result.runtime

    emulation = AnalogEmulatorDataGroup(
        attrs=group_attrs,
        times=Dataset(data=times),
        metrics=metrics_dataset,
        state=state_dataset,
        measurements=measurements_dataset,
    )

    return Datastore(groups={EMULATION_GROUP_KEY: emulation})
