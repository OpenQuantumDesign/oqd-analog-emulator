# 

<p align="center">
  <img src="https://raw.githubusercontent.com/OpenQuantumDesign/oqd-compiler-infrastructure/main/docs/img/oqd-logo-black.png" alt="Open Quantum Design Logo" style="max-height: 200px;">
</p>

<h2 align="center">
    Open Quantum Design: Analog Emulator
</h2>

[![Documentation](https://img.shields.io/badge/documentation-lightblue)](https://docs.openquantumdesign.org/open-quantum-design-analog-emulator)
[![PyPI Version](https://img.shields.io/pypi/v/oqd-analog-emulator)](https://pypi.org/project/oqd-analog-emulator)
[![CI](https://github.com/OpenQuantumDesign/oqd-analog-emulator/actions/workflows/pytest.yml/badge.svg)](https://github.com/OpenQuantumDesign/oqd-analog-emulator/actions/workflows/pytest.yml)
![versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)



## What's here

- [Quick Start](#quickstart) <br/>
- [Installation](#installation) <br/>

## Installation <a name="installation"></a>

To install,
```bash
pip install oqd-analog-emulator
```
or
```bash
pip install git+https://github.com/OpenQuantumDesign/oqd-analog-emulator.git
```

To clone the repository locally for development:

```bash
git clone https://github.com/OpenQuantumDesign/oqd-analog-emulator.git
pip install .
```

This OQD repository depends on the [`oqd-core`](https://github.com/OpenQuantumDesign/oqd-core.git)
and [`oqd-compiler-infrastructure`](https://github.com/OpenQuantumDesign/oqd-compiler-infrastructure.git) packages, which should be installed with all other dependencies.
If needed, these can be manually installed via,
```bash
pip install oqd-compiler-infrastructure
pip install oqd-core
```

## Getting Started <a name="Getting Started"></a>

For example, to simulate a simple Rabi flopping experiment:

```python
import matplotlib.pyplot as plt

from oqd_core.interface.analog.operator import *
from oqd_core.interface.analog.operation import *
from oqd_core.backend.metric import *
from oqd_core.backend.task import Task, TaskArgsAnalog
from oqd_analog_emulator.qutip_backend import QutipBackend

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

### Where in the stack
```mermaid
block-beta
   columns 3

   block:Interface
       columns 1
       InterfaceTitle("<i><b>Interfaces</b><i/>")
       InterfaceDigital["<b>Digital Interface</b>\nQuantum circuits with discrete gates"]
       space
       InterfaceAnalog["<b>Analog Interface</b>\n Continuous-time evolution with Hamiltonians"]
       space
       InterfaceAtomic["<b>Atomic Interface</b>\nLight-matter interactions between lasers and ions"]
       space
    end

    block:IR
       columns 1
       IRTitle("<i><b>IRs</b><i/>")
       IRDigital["Quantum circuit IR\nopenQASM, LLVM+QIR"]
       space
       IRAnalog["openQSIM"]
       space
       IRAtomic["openAPL"]
       space
    end

    block:Emulator
       columns 1
       EmulatorsTitle("<i><b>Classical Emulators</b><i/>")

       EmulatorDigital["Pennylane, Qiskit"]
       space
       EmulatorAnalog["QuTiP, QuantumOptics.jl"]
       space
       EmulatorAtomic["TrICal, QuantumIon.jl"]
       space
    end

    space
    block:RealTime
       columns 1
       RealTimeTitle("<i><b>Real-Time</b><i/>")
       space
       RTSoftware["ARTIQ, DAX, OQDAX"]
       space
       RTGateware["Sinara Real-Time Control"]
       space
       RTHardware["Lasers, Modulators, Photodetection, Ion Trap"]
       space
       RTApparatus["Trapped-Ion QPU (<sup>171</sup>Yt<sup>+</sup>, <sup>133</sup>Ba<sup>+</sup>)"]
       space
    end
    space

   InterfaceDigital --> IRDigital
   InterfaceAnalog --> IRAnalog
   InterfaceAtomic --> IRAtomic

   IRDigital --> IRAnalog
   IRAnalog --> IRAtomic

   IRDigital --> EmulatorDigital
   IRAnalog --> EmulatorAnalog
   IRAtomic --> EmulatorAtomic

   IRAtomic --> RealTimeTitle

   RTSoftware --> RTGateware
   RTGateware --> RTHardware
   RTHardware --> RTApparatus

    classDef title fill:#d6d4d4,stroke:#333,color:#333;
    classDef digital fill:#E7E08B,stroke:#333,color:#333;
    classDef analog fill:#E4E9B2,stroke:#333,color:#333;
    classDef atomic fill:#D2E4C4,stroke:#333,color:#333;
    classDef realtime fill:#B5CBB7,stroke:#333,color:#333;

    classDef highlight fill:#f2bbbb,stroke:#333,color:#333,stroke-dasharray: 5 5;

    class InterfaceTitle,IRTitle,EmulatorsTitle,RealTimeTitle title
    class InterfaceDigital,IRDigital,EmulatorDigital digital
    class InterfaceAnalog,IRAnalog,EmulatorAnalog analog
    class InterfaceAtomic,IRAtomic,EmulatorAtomic atomic
    class RTSoftware,RTGateware,RTHardware,RTApparatus realtime

   class EmulatorAnalog highlight
```
The stack components highlighted in red are contained in this repository.
