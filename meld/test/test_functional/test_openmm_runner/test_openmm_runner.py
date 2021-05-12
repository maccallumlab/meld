#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld import system
from meld.system.openmm_runner import OpenMMRunner
import numpy as np  # type: ignore
import unittest
import os


class TestOpenRunner(unittest.TestCase):
    def setUp(self):
        self.top_path = os.path.join(os.path.dirname(__file__), "system.top")
        self.mdcrd_path = os.path.join(os.path.dirname(__file__), "system.mdcrd")

    def test_implicit_runner(self):
        p = system.ProteinMoleculeFromSequence("NALA ALA CALA")
        b = system.SystemBuilder()
        sys = b.build_system_from_molecules([p])
        sys.temperature_scaler = system.ConstantTemperatureScaler(300.0)

        options = system.RunOptions()
        options.timesteps = 20

        runner = OpenMMRunner(sys, options, platform="Reference")
        state = sys.get_state_template()

        runner.prepare_for_timestep(state, 0.0, 1)
        state = runner.minimize_then_run(state)
        state = runner.run(state)

        assert state

    def test_implicit_runner_amap(self):
        p = system.ProteinMoleculeFromSequence("NALA ALA CALA")
        b = system.SystemBuilder()
        sys = b.build_system_from_molecules([p])
        sys.temperature_scaler = system.ConstantTemperatureScaler(300.0)

        options = system.RunOptions()
        options.timesteps = 20
        options.use_amap = True
        options.amap_beta_bias = 10

        runner = OpenMMRunner(sys, options, platform="Reference")
        state = sys.get_state_template()

        runner.prepare_for_timestep(state, 0.0, 1)
        state = runner.minimize_then_run(state)
        state = runner.run(state)

        assert state

    def test_explicit_runner(self):
        # alanine dipeptide in TIP3P box
        sys = system.builder.load_amber_system(self.top_path, self.mdcrd_path)
        sys.temperature_scaler = system.ConstantTemperatureScaler(300.0)

        options = system.RunOptions(solvation="explicit")
        options.timesteps = 20

        runner = OpenMMRunner(sys, options, platform="Reference")
        state = sys.get_state_template()

        runner.prepare_for_timestep(state, 0.0, 1)
        state = runner.minimize_then_run(state)
        state = runner.run(state)

        assert state

    def test_explicit_runner_scaler(self):
        # alanine dipeptide in TIP3P box
        sys = system.builder.load_amber_system(self.top_path, self.mdcrd_path)
        sys.temperature_scaler = system.ConstantTemperatureScaler(300.0)
        rest2_scaler = system.GeometricTemperatureScaler(0, 1, 300.0, 350.0)

        options = system.RunOptions(solvation="explicit")
        options.rest2_scaler = system.REST2Scaler(300.0, rest2_scaler)
        options.timesteps = 20
        options.use_rest2 = True

        runner = OpenMMRunner(sys, options, platform="Reference")
        state = sys.get_state_template()

        runner.prepare_for_timestep(state, 0.0, 1)
        state = runner.minimize_then_run(state)
        state = runner.run(state)

        assert state
