# ![Open Quantum Design](docs/img/oqd-logo-text.png)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![versions](https://img.shields.io/badge/python-3.11-blue)

<h2 align="center">
    Open Quantum Design: Analog Emulator
</h2>

## What's here

- [Quick Start](#quickstart) <br/>
- [Installation](#installation) <br/>

## Installation <a name="installation"></a>

To install,
```bash
pip install git+https://github.com/OpenQuantumDesign/oqd-analog-emulator.git
```

To clone the repository locally for development:
```bash
git clone https://github.com/OpenQuantumDesign/oqd-analog-emulator.git
pip install .
```

This OQD repository depends on the [`oqd-core`](https://github.com/OpenQuantumDesign/oqd-core.git) 
and [`oqd-compiler-infrastructure`](https://github.com/OpenQuantumDesign/oqd-compiler-infrastructure.git) packages.
```bash
pip install git+https://github.com/OpenQuantumDesign/oqd-compiler-infrastructure.git
pip install git+https://github.com/OpenQuantumDesign/oqd-core.git
```

## Getting Started <a name="Getting Started"></a>
For example, to simulate a simple Rabi flopping experiment:

```python
import matplotlib.pyplot as plt

from core.interface.analog.operator import *
from core.interface.analog.operation import *
from core.backend.metric import *
from core.backend.task import Task, TaskArgsAnalog
from analog_emulator.qutip_backend import QutipBackend

X = PauliX()
Z = PauliZ()

circuit = AnalogCircuit()
circuit.evolve(duration=10, gate=AnalogGate(hamiltonian=X))

args = TaskArgsAnalog(
    n_shots=100,
    fock_cutoff=4,
    metrics={"Z": Expectation(operator=Z)},
    dt=1e-3,
)

task = Task(program=circuit, args=args)

backend = QutipBackend()
results = backend.run(task=task)

plt.plot(results.times, results.metrics["Z"], label=f"$\\langle Z \\rangle$")
```
