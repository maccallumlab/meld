#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld import (
    AmberSubSystemFromSequence,
    AmberSystemBuilder,
    AmberOptions,
    RunOptions,
)
from meld.system import temperature
from meld.runner import openmm_runner
from openmm import unit as u  # type: ignore

import unittest


class TestOpenRunner(unittest.TestCase):
    def test_implicit_runner(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        sys = b.build_system([p]).finalize()
        sys.temperature_scaler = temperature.ConstantTemperatureScaler(300.0 * u.kelvin)

        opt = RunOptions(timesteps=20)

        runner = openmm_runner.OpenMMRunner(sys, opt, platform="Reference")
        runner.prepare_for_timestep(sys.get_state_template(), 0.0, 1)

        s = sys.get_state_template()
        s = runner.minimize_then_run(s)
        s = runner.run(s)

    def test_implicit_runner_amap(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        options = AmberOptions(enable_amap=True)
        b = AmberSystemBuilder(options)
        sys = b.build_system([p]).finalize()
        sys.temperature_scaler = temperature.ConstantTemperatureScaler(300.0 * u.kelvin)

        opt = RunOptions(timesteps=20)

        runner = openmm_runner.OpenMMRunner(sys, opt, platform="Reference")
        runner.prepare_for_timestep(sys.get_state_template(), 0.0, 1)

        s = sys.get_state_template()
        s = runner.minimize_then_run(s)
        s = runner.run(s)

    def test_explicit_runner(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        options = AmberOptions(solvation="explicit", cutoff=1.0, enable_pme=True, enable_pressure_coupling=True)
        b = AmberSystemBuilder(options)
        sys = b.build_system([p]).finalize()
        sys.temperature_scaler = temperature.ConstantTemperatureScaler(300.0 * u.kelvin)

        opt = RunOptions(timesteps=2, minimize_steps=100)

        runner = openmm_runner.OpenMMRunner(sys, opt, platform="Reference")
        runner.prepare_for_timestep(sys.get_state_template(), 0.0, 1)

        s = sys.get_state_template()
        s = runner.minimize_then_run(s)
        s = runner.run(s)

    def test_explicit_runner_scaler(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        options = AmberOptions(
            solvation="explicit", cutoff=1.0, enable_pme=True, enable_pressure_coupling=True
        )
        b = AmberSystemBuilder(options)
        sys = b.build_system([p]).finalize()
        sys.temperature_scaler = temperature.ConstantTemperatureScaler(300.0 * u.kelvin)
        rest2_scaler = temperature.GeometricTemperatureScaler(
            0, 1, 300.0 * u.kelvin, 350.0 * u.kelvin
        )

        opt = RunOptions(
            rest2_scaler=temperature.REST2Scaler(300.0 * u.kelvin, rest2_scaler),
            minimize_steps=100,
            timesteps=2,
            use_rest2=True
        )

        runner = openmm_runner.OpenMMRunner(sys, opt, platform="Reference")
        runner.prepare_for_timestep(sys.get_state_template(), 0.0, 1)

        s = sys.get_state_template()
        s = runner.minimize_then_run(s)
        s = runner.run(s)
