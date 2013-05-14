#!/usr/bin/env python
# encoding: utf-8


from meld import system
import numpy


p = system.ProteinMoleculeFromSequence('NALA ALA CALA')
b = system.SystemBuilder()
sys = b.build_system_from_molecules([p])
sys.temperature_scaler = system.ConstantTemperatureScaler(300.)

runner = system.OpenMMRunner(sys)
runner.set_alpha(0.)
runner.options.timesteps = 10000

pos = sys._coordinates.copy()
vel = numpy.zeros_like(pos)
alpha = 0.
energy = 0.
state = system.SystemState(pos, vel, alpha, energy)

state = runner.minimize_then_run(state)
state = runner.run(state)

