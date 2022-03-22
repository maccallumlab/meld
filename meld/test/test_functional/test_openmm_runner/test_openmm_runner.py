#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld import AmberSubSystemFromSequence, AmberSystemBuilder
from meld.system import temperature
from meld.system import options
from meld.system import state
from meld.runner import openmm_runner
from openmm import unit as u  # type: ignore

import numpy as np  # type: ignore
import unittest
import os


class TestOpenRunner(unittest.TestCase):
    def test_implicit_runner(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        b = AmberSystemBuilder()
        sys = b.build_system([p]).finalize()
        sys.temperature_scaler = temperature.ConstantTemperatureScaler(300.0 * u.kelvin)

        opt = options.RunOptions()
        opt.timesteps = 20

        runner = openmm_runner.OpenMMRunner(sys, opt, platform="Reference")
        runner.prepare_for_timestep(sys.get_state_template(), 0.0, 1)

        s = sys.get_state_template()
        s = runner.minimize_then_run(s)
        s = runner.run(s)

    def test_implicit_runner_amap(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        b = AmberSystemBuilder()
        sys = b.build_system([p], enable_amap=True).finalize()
        sys.temperature_scaler = temperature.ConstantTemperatureScaler(300.0 * u.kelvin)

        opt = options.RunOptions()
        opt.timesteps = 20

        runner = openmm_runner.OpenMMRunner(sys, opt, platform="Reference")
        runner.prepare_for_timestep(sys.get_state_template(), 0.0, 1)

        s = sys.get_state_template()
        s = runner.minimize_then_run(s)
        s = runner.run(s)

    def test_explicit_runner(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        b = AmberSystemBuilder(solvation="explicit")
        sys = b.build_system([p], cutoff=1.0).finalize()
        sys.temperature_scaler = temperature.ConstantTemperatureScaler(300.0 * u.kelvin)

        opt = options.RunOptions()
        opt.solvation = "explicit"
        opt.minimize_steps = 100
        opt.timesteps = 2

        runner = openmm_runner.OpenMMRunner(sys, opt, platform="Reference")
        runner.prepare_for_timestep(sys.get_state_template(), 0.0, 1)

        s = sys.get_state_template()
        s = runner.minimize_then_run(s)
        s = runner.run(s)

    def test_explicit_runner_scaler(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        b = AmberSystemBuilder(solvation="explicit")
        sys = b.build_system([p], cutoff=1.0).finalize()
        sys.temperature_scaler = temperature.ConstantTemperatureScaler(300.0 * u.kelvin)
        rest2_scaler = temperature.GeometricTemperatureScaler(
            0, 1, 300.0 * u.kelvin, 350.0 * u.kelvin
        )

        opt = options.RunOptions()
        opt.solvation = "explicit"
        opt.rest2_scaler = temperature.REST2Scaler(300.0 * u.kelvin, rest2_scaler)
        opt.minimize_steps = 100
        opt.timesteps = 2
        opt.use_rest2 = True

        runner = openmm_runner.OpenMMRunner(sys, opt, platform="Reference")
        runner.prepare_for_timestep(sys.get_state_template(), 0.0, 1)

        s = sys.get_state_template()
        s = runner.minimize_then_run(s)
        s = runner.run(s)
