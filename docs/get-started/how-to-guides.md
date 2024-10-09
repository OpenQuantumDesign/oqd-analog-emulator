# How-to: A hands on guide

<!-- prettier-ignore -->
/// admonition | Note
    type: note

This part of the project documentation focuses on a **problem-oriented** approach. You'll tackle common tasks that you might have, with the help of the code provided in this project.
///

## Analog

Let's implement our favourite Hamiltonian -- the transverse-field Ising model.
The general Hamiltonian looks like,

$$
H = \sum_{\langle ij \rangle} \sigma^x_i \sigma^x_j + h \sum_i \sigma^z_i
$$

Let's implement it with two qubits and with $h=1$.

$$
H = \sigma^x_1 \sigma^x_2 + \sigma^z_1 + \sigma^z_2
$$

Our analog circuit will have one gate, which describes this Hamiltonian.

```py
from midstack.analog import AnalogCircuit, AnalogGate, PauliX, PauliZ, PauliI

circuit = AnalogCircuit()
circuit.evolve(
    AnalogGate(
        duration=1.0,
        hamiltonian=[PauliX @ PauliX, PauliZ @ PauliI, PauliI @ PauliZ],
    )
)
```

Let's now generalize this to a quantum system with `n` qubits.

```py
from midstack.analog import AnalogCircuit, AnalogGate, PauliX, PauliZ, PauliI
from midstack.analog.math import tensor

n = 10
circuit = AnalogCircuit()

field = [tensor([PauliZ if i == j else PauliI for i in range(n)]) for j in range(n)]
interaction = [tensor([PauliX if i in (i, (i+1)%n) else PauliI for i in range(n)])]
hamiltonian = interaction + field

circuit.evolve(
    AnalogGate(
        duration=1.0,
        hamiltonian=hamiltonian
    )
)
```

We will emulate this quantum evolution on two classical backends.
The first is a wrapper around Qutip, the second a wrapper around the QuantumOptics.jl package.

```py
from backends.analog.python.qutip import QutipBackend
from backends.analog.julia.quantumoptics import QuantumOpticsBackend
from backends.task import Task, TaskArgsAnalog

args = TaskArgsAnalog(
    n_shots=100,
    fock_cutoff=4,
    dt=0.01,
)
task = Task(program=circuit, args=args)
backend = QutipBackend()
result = backend.run(task)
```

```py
import matplotlib.pyplot as plt
from backends.metric import Expectation, EntanglementEntropyVN

metrics = {
    'entanglement_entropy': EntanglementEntropyVN(qreg=[i for i in range(n//2)]),
    'expectation_z': Expectation(operator=field)
}
args = TaskArgsAnalog(
    n_shots=100,
    fock_cutoff=4,
    metrics=metrics,
)
task = Task(program=circuit, args=args)
backend = QutipBackend()
result = backend.run(task)

plt.plot(result.times, result.metrics['entanglement_entropy'])

```

![Entropy of entanglement](../img/plots/entropy_entanglement.png)

## Digital

## Atomic
