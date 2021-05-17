#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld.system.subsystem import SubSystemFromSequence
from meld.system.builder import SystemBuilder
from meld.system.temperature import ConstantTemperatureScaler, GeometricTemperatureScaler, REST2Scaler
from meld.system.options import RunOptions
from meld.system.state import SystemState
from meld.system.openmm_runner import OpenMMRunner
import numpy as np  # type: ignore
import unittest
import os


class TestOpenRunner(unittest.TestCase):
    def setUp(self):
        self.top_path = os.path.join(os.path.dirname(__file__), "system.top")
        self.mdcrd_path = os.path.join(os.path.dirname(__file__), "system.mdcrd")

    def test_implicit_runner(self):
        p = SubSystemFromSequence("NALA ALA CALA")
        b = SystemBuilder()
        sys = b.build_system([p])
        sys.temperature_scaler = ConstantTemperatureScaler(300.)

        options = RunOptions()
        options.timesteps = 20

        runner = OpenMMRunner(sys, options, platform="Reference")
        runner.prepare_for_timestep(0., 1)

        pos = sys._coordinates.copy()
        vel = np.zeros_like(pos)
        alpha = 0.
        energy = 0.
        box_vectors = np.zeros(3)
        state = SystemState(pos, vel, alpha, energy, box_vectors)

        state = runner.minimize_then_run(state)
        state = runner.run(state)

        assert state

    def test_implicit_runner_amap(self):
        p = SubSystemFromSequence("NALA ALA CALA")
        b = SystemBuilder()
        sys = b.build_system([p])
        sys.temperature_scaler = ConstantTemperatureScaler(300.)

        options = RunOptions()
        options.timesteps = 20
        options.use_amap = True
        options.amap_beta_bias = 10

        runner = OpenMMRunner(sys, options, platform="Reference")
        runner.prepare_for_timestep(0., 1)

        pos = sys._coordinates.copy()
        vel = np.zeros_like(pos)
        alpha = 0.
        energy = 0.
        box_vectors = np.zeros(3)
        state = SystemState(pos, vel, alpha, energy, box_vectors)

        state = runner.minimize_then_run(state)
        state = runner.run(state)

        assert state

    def test_explicit_runner(self):
        p = SubSystemFromSequence("NALA ALA CALA")
        b = SystemBuilder(explicit_solvent=True)
        sys = b.build_system([p])
        sys.temperature_scaler = ConstantTemperatureScaler(300.)

        options = RunOptions(solvation="explicit")
        options.minimize_steps = 100
        options.timesteps = 2

        runner = OpenMMRunner(sys, options, platform="Reference")
        runner.prepare_for_timestep(0., 1)

        pos = sys._coordinates.copy()
        vel = np.zeros_like(pos)
        alpha = 0.
        energy = 0.
        box_vectors = sys._box_vectors
        state = SystemState(pos, vel, alpha, energy, box_vectors)

        state = runner.minimize_then_run(state)
        state = runner.run(state)

        assert state

    def test_explicit_runner_scaler(self):
        p = SubSystemFromSequence("NALA ALA CALA")
        b = SystemBuilder(explicit_solvent=True)
        sys = b.build_system([p])
        sys.temperature_scaler = ConstantTemperatureScaler(300.)
        rest2_scaler = GeometricTemperatureScaler(0, 1, 300., 350.)

        options = RunOptions(solvation="explicit")
        options.rest2_scaler = REST2Scaler(300., rest2_scaler)
        options.minimize_steps = 100
        options.timesteps = 2
        options.use_rest2 = True

        runner = OpenMMRunner(sys, options, platform="Reference")
        runner.prepare_for_timestep(0., 1)

        pos = sys._coordinates.copy()
        vel = np.zeros_like(pos)
        alpha = 0.
        energy = 0.
        box_vectors = sys._box_vectors
        state = SystemState(pos, vel, alpha, energy, box_vectors)

        state = runner.minimize_then_run(state)
        state = runner.run(state)

        assert state
