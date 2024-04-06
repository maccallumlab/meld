"""
Package for running MELD simulations in OpenMM
"""

from typing import Optional

from meld import interfaces
from meld.system import options


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
        import meld.runner.openmm_runner

        return meld.runner.openmm_runner.OpenMMRunner(
            system, options, communicator=comm, platform=platform
        )
    elif options.runner == "fake_runner":
        import meld.runner.fake_runner

        return meld.runner.fake_runner.FakeSystemRunner(system, options, comm)
    else:
        raise RuntimeError(f"Unknown type of runner: {options.runner}")
