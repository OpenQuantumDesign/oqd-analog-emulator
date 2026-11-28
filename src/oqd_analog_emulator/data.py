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
from datetime import datetime

import numpy as np
from oqd_dataschema import (
    Dataset,
    Datastore,
    Folder,
    GroupBase,
    unstructured_to_structured,
)


class RegisterGroup(GroupBase):
    reg_name: Folder
    parts: Folder
    state: Dataset
    time: Dataset
    time_last_updated: Dataset

class MetricGroup(GroupBase):
    key: Folder
    value: Dataset

class HistoryGroup(GroupBase):
    parts: Folder
    state: Dataset
    time: Dataset
    time_last_updated: Dataset


def create_reg_group(reg_name, parts, state, time, time_last_updated):
    registers_group = RegisterGroup(
        reg_name=Folder(
            document_schema=dict(
                index="int32",
                dim="int32",
            ),
            data=unstructured_to_structured(
                [reg_name],
                dtype=np.dtype(
                    [
                        ("index", "int32"),
                        ("dim", "int32"),
                    ]
                ),
            ),
        ),
        parts=Folder(
            document_schema=dict(
                index="int32",
                dim="int32",
            ),
            data=unstructured_to_structured(
                np.array(parts, dtype=np.int32),
                dtype=np.dtype(
                    [
                        ("index", "int32"),
                        ("dim", "int32")
                    ]
                ),
            ),
        ),
        state=Dataset(
            data=state
        ),
        time=Dataset(
            data=np.array([time], dtype=np.float32),
        ),
        time_last_updated=Dataset(
            data=np.array([time_last_updated], dtype=np.float32),
        ),
    )
    return registers_group

def create_metric_group(key, value):
    metrics_group = MetricGroup(
        key = Folder(
            document_schema=dict(
                name="str",
                operator="str",
                target="str",
            ),
            data=np.array(
                [key],
                dtype = np.dtype(
                    [
                        ("name", np.dtype("<U10")),
                        ("operator", np.dtype("<U10")),
                        ("target", np.dtype("<U10")),
                    ]
                ),
            ),
        ),
        value = Dataset(
            data=np.array(value, dtype=np.float64),
        ),
    )
    return metrics_group

def create_hist_group(parts, state, time, time_last_updated):
    history_group = HistoryGroup(
        parts=Folder(
            document_schema=dict(
                index="int32",
                dim="int32",
            ),
            data=unstructured_to_structured(
                np.array(parts, dtype=np.int32),
                dtype=np.dtype(
                    [
                        ("index", "int32"),
                        ("dim", "int32")
                    ]
                ),
            ),
        ),
        state=Dataset(
            data=state
        ),
        time=Dataset(
            data=np.array([time], dtype=np.float32),
        ),
        time_last_updated=Dataset(
            data=np.array([time_last_updated], dtype=np.float32),
        ),
    )
    return history_group

def write_to_hdf5(register, _store, file_name = "interp_state"):
    _registers = register._registers
    _history = register.history
    _metrics = register.metrics

    n = 0
    groups = {}
    creation_time = datetime.now()

    for k, v in _registers.items():
        temp_parts = []
        for part in v.parts:
            temp_parts.append([part.index, part.dim])

        try:
            state = v.state.data_as("ndarray")
        except ValueError:
            state = v.state.data_as("csr_array").toarray()
        except Exception as e:
            print(f"Error while getting the state: {e}")
            state = np.array([])

        var_name = f"register_{n}"
        groups[var_name] = create_reg_group(
            reg_name=[k.index, k.dim],
            parts=temp_parts,
            state=state,
            time=v.time,
            time_last_updated=v.time_last_updated
        )
        n += 1
    
    n = 0
    for k, v in _metrics.items():
        var_name = f"metric_{n}"
        groups[var_name] = create_metric_group(k, v)
        n += 1
    
    n = 0
    for k, v in _history.items():
        for i, reg in enumerate(v):
            temp_parts = []
            for part in reg.parts:
                temp_parts.append([part.index, part.dim])

            if reg.state is not None:
                try:
                    state = reg.state.data_as("ndarray")
                except ValueError:
                    state = reg.state.data_as("csr_array").toarray()
                except Exception as e:
                    print(f"Error while getting the state: {e}")
                    state = np.array([])
            else:
                state = np.array([])

            var_name = f"history_{n}_reg_{i}"
            groups[var_name] = create_hist_group(
                parts=temp_parts,
                state=state,
                time=reg.time,
                time_last_updated=reg.time_last_updated
            )
        n += 1

    store = Datastore(groups=groups, attrs={"created_at": f"{creation_time}"},)

    file_name += f"_{creation_time}"

    store.model_dump_hdf5(pathlib.Path(f"{file_name}.h5"))

