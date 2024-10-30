# ![Open Quantum Design](./img/oqd-logo-text.png)

<div align="center">
    <h2 align="center">
        Open Quantum Design: Analog Emulators
    </h2>
</div>

![Python](https://img.shields.io/badge/Python-3.10_|_3.11_|_3.12-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
<!-- prettier-ignore -->
/// admonition | Note
    type: note
Welcome to Open Quantum Design.
This documentation is still under development, we welcome contributions! © Open Quantum Design
///


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
   
   classDef title fill:#d6d4d4,stroke:#333;
   classDef digital fill:#E7E08B,stroke:#333;
   classDef analog fill:#E4E9B2,stroke:#333;
   classDef atomic fill:#D2E4C4,stroke:#333;
   classDef realtime fill:#B5CBB7,stroke:#333;

    classDef highlight fill:#f2bbbb,stroke:#333,stroke-dasharray: 5 5;

    class InterfaceTitle,IRTitle,EmulatorsTitle,RealTimeTitle title
    class InterfaceDigital,IRDigital,EmulatorDigital digital
    class InterfaceAnalog,IRAnalog,EmulatorAnalog analog
    class InterfaceAtomic,IRAtomic,EmulatorAtomic atomic
    class RTSoftware,RTGateware,RTHardware,RTApparatus realtime
   
   class EmulatorAnalog highlight
```
The stack components highlighted in red are contained in this repository.