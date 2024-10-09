from ..base import BackendBase
from .interface import QutipExperiment, TaskArgsQutip
from ..task import Task

########################################################################################

__all__ = [
    "QutipBackend",
]

########################################################################################


class QutipBackend(BackendBase):
    """
    Class representing the Qutip backend
    """

    def compile(self, task: Task):
        """
        Method for compiling program of task to a [`QutipExperiment`][midstack.backend.qutip.interface.QutipExperiment] and converting
        args of task to [`TaskArgsQutip`][midstack.backend.qutip.interface.TaskArgsQutip].

        Args:
            task (Task): Quantum experiment to compile

        Returns:
            converted_circuit (QutipExperiment): QutipExperiment containing the compiled experiment for Qutip
            converted_args (TaskArgsQutip): args of analog layer are converted to args for QuTip.

        """
        from midstack.compiler.analog.passes.canonicalize import (
            analog_operator_canonicalization,
        )
        from midstack.compiler.analog.passes.assign import (
            assign_analog_circuit_dim,
            verify_analog_args_dim,
        )
        from midstack.backend.qutip.passes import (
            compiler_analog_args_to_qutipIR,
            compiler_analog_circuit_to_qutipIR,
        )

        # pass to canonicaliza the operators in the AnalogCircuit
        canonicalized_circuit = analog_operator_canonicalization(task.program)

        # This just canonicalizes the operators inside the TaskArgsAnalog
        # i.e. operators for Expectation
        canonicalized_args = analog_operator_canonicalization(task.args)

        # another pass which assigns the n_qreg and n_qmode of the
        # AnalogCircuit IR
        assigned_circuit = assign_analog_circuit_dim(canonicalized_circuit)

        # This just verifies that the operators in the args have the same
        # dimension as the operators in the AnalogCircuit
        verify_analog_args_dim(
            canonicalized_args,
            n_qreg=assigned_circuit.n_qreg,
            n_qmode=assigned_circuit.n_qmode,
        )

        # another pass which compiles AnalogCircuit to a QutipExperiment
        converted_circuit = compiler_analog_circuit_to_qutipIR(
            assigned_circuit, fock_cutoff=task.args.fock_cutoff
        )

        # This just converts the args so that the operators of the args are
        # converted to qutip objects
        converted_args = compiler_analog_args_to_qutipIR(canonicalized_args)

        return (
            converted_circuit,
            converted_args,
        )

    def run(
        self,
        *,
        task: Task = None,
        experiment: QutipExperiment = None,
        args: TaskArgsQutip = None,
    ):
        """
        Method to simulate an experiment using theQuTip backend

        Args:
            task (Optional[Task]): Run experiment from a [`Task`][midstack.backend.task.Task] object
            experiment (Optional[QutipExperiment]): Run experiment from a [`QutipExperiment`][midstack.backend.qutip.interface.QutipExperiment] object

        Returns:
            TaskResultAnalog object containing the simulation results.

        Note:
            only one of task or experiment must provided
        """
        from midstack.backend.qutip.passes import run_qutip_experiment

        if experiment is None and args is not None:
            raise TypeError("args provided without QuTip experiment")
        if experiment is not None and args is None:
            raise TypeError("QuTip experiment provided without args")

        if task is not None and experiment is not None:
            raise TypeError("Both task and experiment are given as inputs to run")
        if experiment is None:
            experiment, args = self.compile(task=task)
        return run_qutip_experiment(model=experiment, args=args)
