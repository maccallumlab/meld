#!/usr/bin/env python
# encoding: utf-8

#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld import system
import numpy as np
import unittest


class TestOpenRunner(unittest.TestCase):
    def test_runner(self):
        p = system.ProteinMoleculeFromSequence('NALA ALA CALA')
        b = system.SystemBuilder()
        sys = b.build_system_from_molecules([p])
        sys.temperature_scaler = system.ConstantTemperatureScaler(300.)

        options = system.RunOptions()
        options.timesteps = 10000
        options.use_amap = True
        options.amap_beta_bias = 10

        runner = system.OpenMMRunner(sys, options, test=True)
        runner.prepare_for_timestep(0., 1)

        pos = sys._coordinates.copy()
        vel = np.zeros_like(pos)
        alpha = 0.
        energy = 0.
        box_vectors = np.zeros(3)
        state = system.SystemState(pos, vel, alpha, energy, box_vectors)

        state = runner.minimize_then_run(state)
        state = runner.run(state)

        assert state
