#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld.system import subsystem
from meld.system import builder
from meld.system import temperature
from meld.system import options
from meld.system import state
from meld.runner import openmm_runner
from openmm import unit as u  # type: ignore

import numpy as np  # type: ignore
import unittest
import os


class TestOpenRunner(unittest.TestCase):
    def setUp(self):
        self.top_path = os.path.join(os.path.dirname(__file__), "system.top")
        self.mdcrd_path = os.path.join(os.path.dirname(__file__), "system.mdcrd")

    def test_implicit_runner(self):
        p = subsystem.SubSystemFromSequence("NALA ALA CALA")
        b = builder.SystemBuilder()
        sys = b.build_system([p])
        sys.temperature_scaler = temperature.ConstantTemperatureScaler(300.0 * u.kelvin)

        opt = options.RunOptions()
        opt.timesteps = 20

        runner = openmm_runner.OpenMMRunner(sys, opt, platform="Reference")
        runner.prepare_for_timestep(sys.get_state_template(), 0.0, 1)

        pos = sys._coordinates.copy()
        vel = np.zeros_like(pos)
        alpha = 0.0
        energy = 0.0
        box_vectors = np.zeros(3)
        s = state.SystemState(pos, vel, alpha, energy, box_vectors)

        s = runner.minimize_then_run(s)
        s = runner.run(s)

        assert s

    def test_implicit_runner_amap(self):
        p = subsystem.SubSystemFromSequence("NALA ALA CALA")
        b = builder.SystemBuilder()
        sys = b.build_system([p])
        sys.temperature_scaler = temperature.ConstantTemperatureScaler(300.0 * u.kelvin)

        opt = options.RunOptions()
        opt.timesteps = 20
        opt.use_amap = True
        opt.amap_beta_bias = 10

        runner = openmm_runner.OpenMMRunner(sys, opt, platform="Reference")
        runner.prepare_for_timestep(sys.get_state_template(), 0.0, 1)

        pos = sys._coordinates.copy()
        vel = np.zeros_like(pos)
        alpha = 0.0
        energy = 0.0
        box_vectors = np.zeros(3)
        s = state.SystemState(pos, vel, alpha, energy, box_vectors)

        s = runner.minimize_then_run(s)
        s = runner.run(s)

        assert s

    def test_explicit_runner(self):
        p = subsystem.SubSystemFromSequence("NALA ALA CALA")
        b = builder.SystemBuilder(explicit_solvent=True)
        sys = b.build_system([p])
        sys.temperature_scaler = temperature.ConstantTemperatureScaler(300.0 * u.kelvin)

        opt = options.RunOptions(solvation="explicit")
        opt.minimize_steps = 100
        opt.timesteps = 2

        runner = openmm_runner.OpenMMRunner(sys, opt, platform="Reference")
        runner.prepare_for_timestep(sys.get_state_template(), 0.0, 1)

        pos = sys._coordinates.copy()
        vel = np.zeros_like(pos)
        alpha = 0.0
        energy = 0.0
        box_vectors = sys._box_vectors
        s = state.SystemState(pos, vel, alpha, energy, box_vectors)

        s = runner.minimize_then_run(s)
        s = runner.run(s)

        assert s

    def test_explicit_runner_scaler(self):
        p = subsystem.SubSystemFromSequence("NALA ALA CALA")
        b = builder.SystemBuilder(explicit_solvent=True)
        sys = b.build_system([p])
        sys.temperature_scaler = temperature.ConstantTemperatureScaler(300.0 * u.kelvin)
        rest2_scaler = temperature.GeometricTemperatureScaler(
            0, 1, 300.0 * u.kelvin, 350.0 * u.kelvin
        )

        opt = options.RunOptions(solvation="explicit")
        opt.rest2_scaler = temperature.REST2Scaler(300.0 * u.kelvin, rest2_scaler)
        opt.minimize_steps = 100
        opt.timesteps = 2
        opt.use_rest2 = True

        runner = openmm_runner.OpenMMRunner(sys, opt, platform="Reference")
        runner.prepare_for_timestep(sys.get_state_template(), 0.0, 1)

        pos = sys._coordinates.copy()
        vel = np.zeros_like(pos)
        alpha = 0.0
        energy = 0.0
        box_vectors = sys._box_vectors
        s = state.SystemState(pos, vel, alpha, energy, box_vectors)

        s = runner.minimize_then_run(s)
        s = runner.run(s)

        assert s
