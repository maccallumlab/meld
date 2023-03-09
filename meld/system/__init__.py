#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Module for interacting with MELD systems.

The primary classes are:

- :class:`meld.system.meld_system.System` is the main class that describes a MELD system.
  Systems are built using the builder objects described below. Once built, :class:`System`
  objects may have a variety of restraints added.
- :class:`meld.system.options.RunOptions` is a class that specifies options for a MELD run.
- :class:`meld.system.state.SystemState` is a class that represents the current state
  of a MELD run.

The main classes to build a system are:

- :class:`meld.system.builders.amber.subsystem.AmberSubSystemFromSequence` is used to build a sub-system starting
  from a sequence.
- :class:`meld.system.builders.amber.subsystem.AmberSubSystemFromPdbFile` is used to build a sub-system from a 
  PDB file.
- :class:`meld.system.builders.amber.builder.AmberSystemBuilder` is used to combine SybSystems together into a
  system.

There are a few options for how to couple the temperature to the value of alpha:

- :class:`meld.system.temperature.ConstantTemperatureScaler`
- :class:`meld.system.temperature.LinearTemperatureScaler`
- :class:`meld.system.temperature.GeometricTemperatureScaler`
- :class:`meld.system.temperature.REST2Scaler` for explicit solvent
"""
