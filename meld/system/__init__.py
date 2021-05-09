#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld.system.protein import ProteinMoleculeFromSequence, ProteinMoleculeFromPdbFile
from meld.system.builder import SystemBuilder
from meld.system.system import (
    System,
    ConstantTemperatureScaler,
    LinearTemperatureScaler,
    FixedTemperatureScaler,
    GeometricTemperatureScaler,
    REST2Scaler,
    RunOptions,
)
from meld.system.state import SystemState


def get_runner(system, options, comm, platform=None):
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
