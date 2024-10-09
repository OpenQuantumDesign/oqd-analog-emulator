from midstack.backend.qutip.conversion import (
    QutipBackendCompiler,
    QutipExperimentVM,
    QutipMetricConversion,
)

from oqd_compiler_infrastructure import Post, Pre

########################################################################################

__all__ = [
    "compiler_analog_circuit_to_qutipIR",
    "compiler_analog_args_to_qutipIR",
    "run_qutip_experiment",
]

########################################################################################


def compiler_analog_circuit_to_qutipIR(model, fock_cutoff):
    """
    This compiles ([`AnalogCircuit`][midstack.interface.analog.operations.AnalogCircuit] to a list of  [`QutipOperation`][midstack.backend.qutip.interface.QutipOperation] objects

    Args:
        model (AnalogCircuit):
        fock_cutoff (int): fock_cutoff for Ladder Operators

    Returns:
        (list(QutipOperation)):

    """
    return Post(QutipBackendCompiler(fock_cutoff=fock_cutoff))(model=model)


def compiler_analog_args_to_qutipIR(model):
    """
    This compiles TaskArgsAnalog to a list of TaskArgsQutip


    Args:
        model (TaskArgsAnalog):

    Returns:
        (TaskArgsQutip):

    """
    return Post(QutipBackendCompiler(fock_cutoff=model.fock_cutoff))(model=model)


def run_qutip_experiment(model, args):
    """
    This takes in a [`QutipExperiment`][midstack.backend.qutip.interface.QutipExperiment] and produces a TaskResultAnalog object

    Args:
        model (QutipExperiment):

    Returns:
        (TaskResultAnalog): Contains results of the simulation

    """
    n_qreg = model.n_qreg
    n_qmode = model.n_qmode
    metrics = Post(QutipMetricConversion(n_qreg=n_qreg, n_qmode=n_qmode))(args.metrics)
    interpreter = Pre(
        QutipExperimentVM(
            qt_metrics=metrics,
            n_shots=args.n_shots,
            fock_cutoff=args.fock_cutoff,
            dt=args.dt,
        )
    )
    interpreter(model=model)

    return interpreter.children[0].results
