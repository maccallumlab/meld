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
)
from meld.system.system import GeometricTemperatureScaler, REST2Scaler, RunOptions
from meld.system.state import SystemState


def get_runner(system, options, comm):
    if options.runner == "openmm":
        import meld.system
        return meld.system.openmm_runner.OpenMMRunner(system, options, comm)
    elif options.runner == "fake_runner":
        from meld.system.runner import FakeSystemRunner
        return FakeSystemRunner(system, options, comm)
    else:
        raise RuntimeError(f"Unknown type of runner: {options.runner}")
