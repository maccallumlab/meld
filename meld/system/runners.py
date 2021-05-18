"""
Module for getting the runner for a system
"""

from meld import interfaces
from meld.system import options

from typing import Optional


def get_runner(
    system: interfaces.ISystem,
    options: options.RunOptions,
    comm: Optional[interfaces.ICommunicator],
    platform: str,
):
    """
    Get the runner for the system

    Args:
        system: system to run
        options: options for run
        comm: communicator for run
        platorm: platform to run on [Reference, CPU, CUDA]

    Returns:
        the runner
    """
    if options.runner == "openmm":
        import meld.system.openmm_runner

        return meld.system.openmm_runner.runner.OpenMMRunner(
            system, options, communicator=comm, platform=platform
        )
    elif options.runner == "fake_runner":
        return FakeSystemRunner(system, options, comm)
    else:
        raise RuntimeError(f"Unknown type of runner: {options.runner}")


class FakeSystemRunner(interfaces.IRunner):
    """
    Fake runner for test purposes.
    """

    def __init__(
        self,
        system,
        options,
        communicator,
    ) -> None:
        self.temperature_scaler = system.temperature_scaler

    def set_alpha_and_timestep(self, alpha: float, timestep: int) -> None:
        pass

    def minimize_then_run(self, state: interfaces.IState) -> interfaces.IState:
        return state

    def run(self, state: interfaces.IState) -> interfaces.IState:
        return state

    def get_energy(self, state: interfaces.IState) -> float:
        return 0.0

    def prepare_for_timestep(self, alpha: float, timestep: int) -> None:
        pass
