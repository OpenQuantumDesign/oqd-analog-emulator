# Single qubit Rabi flopping

Let's implement the single qubit Rabi Flopping,

$$
H = -\frac{\pi}{4}\sigma^x
$$

## Implementation

We will go through this step by step. First we get the necessary imports:
/// details | Imports

```py
from rich import print as pprint

import numpy as np

from core.interface.analog.operator import *
from core.interface.analog.operation import *
from core.backend.metric import *
from core.backend.task import Task, TaskArgsAnalog
from analog_emulator.qutip_backend import QutipBackend
```

///

Then we define the [`AnalogGate`][core.interface.analog.operations.AnalogGate] object

```py
"""For simplicity we initialize the X Operator"""
X = PauliX()

H = AnalogGate(hamiltonian= -(np.pi / 4) * X)
```

Then we define the [`AnalogCircuit`][core.interface.analog.operations.AnalogCircuit] object and evolve it according to the hamiltonian defined above

```py
ac = AnalogCircuit()
ac.evolve(duration=3, gate=H)
```

For QuTip simulation we need to define the arguements which contain the number of shots and the metrics we want to evaluate.

```py
args = TaskArgsAnalog(
    n_shots=100,
    fock_cutoff=4,
    metrics={
        "Z": Expectation(operator = Z),
    },
    dt=1e-3,
)
```

We can then wrap the [`AnalogCircuit`][core.interface.analog.operations.AnalogCircuit] and the args to a [`Task`][core.backend.task.Task] object and run using the QuTip backend. Note that there are 2 ways to run and the 2 ways are explained.

## Running the simulation

First initialize the [`QutipBackend`][core.backend.qutip.base.QutipBackend] object.
=== "Compile & Simulate"
The [`Task`][core.backend.task.Task] can be compiled first to a [`QutipExperiment`][core.backend.qutip.interface.QutipExperiment] object and then this [`QutipExperiment`][core.backend.qutip.interface.QutipExperiment] object can be run. This is to allow you to see what parameters are used to specify the particular QuTip experiment.

    ``` py
    backend = QutipBackend()
    experiment = backend.compile(task = task)
    results = backend.run(experiment = experiment)
    ```

=== "Directly Simulate"
The [`Task`][core.backend.task.Task] object can be directly simulated by the `run()` method.

    ``` py
    backend = QutipBackend()
    results = backend.run(task = task)
    ```

