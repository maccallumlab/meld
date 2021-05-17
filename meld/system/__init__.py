#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Module for interacting with MELD systems.

The primary classes are:

- :class:`meld.system.system.System` is the main class that describes a MELD system.
  Systems are built using the builder objects described below. Once built, :class:`System`
  objects may have a variety of restraints added.
- :class:`meld.system.options.RunOptions` is a class that specifies options for a MELD run.
- :class:`meld.system.state.SystemState` is a class that represents the current state
  of a MELD run.

The main classes to build a system are:

- :class:`meld.system.subsystem.SubSystemFromSequence` is used to build a sub-system starting
  from a sequence.
- :class:`meld.system.subsystem.SubSystemFromPdbFile` is used to build a sub-system from a 
  PDB file.
- :class:`meld.system.builder.SystemBuilder` is used to combine SybSystems together into a
  system.

There are a few options for how to couple the temperature to the value of alpha:

- :class:`meld.system.temperature.ConstantTemperatureScaler`
- :class:`meld.system.temperature.LinearTemperatureScaler`
- :class:`meld.system.temperature.GeometricTemperatureScaler`
- :class:`meld.system.temperature.REST2Scaler` for explicit solvent
"""

def get_runner(
    system,
    options,
    comm,
    platform,
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
        import meld.system
        import meld.system.openmm_runner

        return meld.system.openmm_runner.runner.OpenMMRunner(
            system, options, communicator=comm, platform=platform
        )
    elif options.runner == "fake_runner":
        from meld.system.runner import FakeSystemRunner

        return FakeSystemRunner(system, options, comm)
    else:
        raise RuntimeError(f"Unknown type of runner: {options.runner}")
