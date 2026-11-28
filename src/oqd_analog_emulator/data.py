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
from typing import Dict

import numpy as np
from oqd_dataschema import (
    Attrs,
    CastDataset,
    CastFolder,
    Dataset,
    Datastore,
    Folder,
    GroupBase,
    GroupRegistry,
    Table,
    condataset,
    unstructured_to_structured,
)
from pydantic import Field


class RegisterGroup(GroupBase):
    attrs: Attrs = Field(
        default_factory=lambda: dict(
            timestamp=str(datetime.datetime.now(datetime.timezone.utc))
        )
    )
    # document: Dict[str, CastDataset]
    reg_name: Folder
    parts: Folder
    state: Dataset
    time: Dataset
    time_last_updated: Dataset

def create_reg_group(reg_name, parts, state, time, time_last_updated):
    creation_time = datetime.now()
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
        attrs={"created_at": f"{creation_time}"},
    )
    return registers_group

def write_to_hdf5(register, _store):
    _registers = register._registers
    _history = register._history
    _metrics = register._metrics

    n = 0
    groups = {}

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
        groups[var_name] = create_reg_group(reg_name=[k.index, k.dim], parts=temp_parts, state=state, time=v.time, time_last_updated=v.time_last_updated)
        n += 1
    
    store = Datastore(groups=groups)
    
    # print(store)

    
    print("metrics: ", _metrics)
    # print("history: ", _history)

    

            # time=Dataset(
            #     data=np.array(collect_time, dtype=np.float32),
            # ),
            # time_last_updated=Dataset(
            #     data=np.array(collect_time_last_updated, dtype=np.float32),
            # ),
            # state=Dataset(
            #     data=np.array(collect_state),
            # ),
            # metrics=Dataset(
            #     data=_metrics,
            # ),
            # store=Dataset(
            #     data=_store,
            # ),
        #     state=Folder(
        #     document_schema=dict(
        #         state="complex128",
        #     ),
        #     data=unstructured_to_structured(
        #         collect_state[0],
        #         dtype=np.dtype(
        #             [
        #                 ("state", "complex128"),
        #             ]
        #         ),
        #     ),
        # ),
        # # ),
    

    store.model_dump_hdf5(pathlib.Path("test.h5"))

