## Compilation

We first compile [`AnalogCircuit`][core.interface.analog.operation.AnalogCircuit] to a [`QutipExperiment`][core.backend.qutip.interface.QutipExperiment] using the ConversionRule [`QutipBackendCompiler`][core.backend.qutip.conversion.QutipBackendCompiler]. We also then separately convert the args of Analog layer to a corresponding representation of the args which can be processed by QuTip. The [`compile`][core.backend.qutip.base.QutipBackend.compile] functions is used to compile [`AnalogCircuit`][core.interface.analog.operation.AnalogCircuit] to a [`QutipExperiment`][core.backend.qutip.interface.QutipExperiment] and also used to convert the args.

## Simulation

After compilation, the actual experiment is run through a Virtual Machine. In classical compilers Virtual Machines takes in list of instructions and runs them. Similarly, the QutipExperiment object has an `instructions` attribute which is a list of [`QutipOperation`][core.backend.qutip.interface.QutipOperation] that can be run through qutip's sesolve function.

```mermaid
    stateDiagram-v2
    User_Input --> OpenQSIM: The user inputs are put in the OpenQSIM AST's Task object
    OpenQSIM --> OpenQSIM(canonicalized): RewriteRule (Canonicalization of Operators)
    OpenQSIM(canonicalized) --> QutipExperiment: compile the program of Task to QutipExperiment. And also convert the analog
    QutipExperiment --> TaskResultAnalog: Run the experiment using the QutipExperimentVM Virtual Machine.
```

For examples you can look [here](\examples)
