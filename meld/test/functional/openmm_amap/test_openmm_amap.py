#!/usr/bin/env python
# encoding: utf-8

from meld import system
import numpy


def main():
    p = system.ProteinMoleculeFromSequence('NALA ALA CALA')
    b = system.SystemBuilder()
    sys = b.build_system_from_molecules([p])
    sys.temperature_scaler = system.ConstantTemperatureScaler(300.)

    options = system.RunOptions()
    options.timesteps = 10000
    options.use_amap = True
    options.amap_beta_bias = 10

    runner = system.OpenMMRunner(sys, options)
    runner.set_alpha(0.)

    pos = sys._coordinates.copy()
    vel = numpy.zeros_like(pos)
    alpha = 0.
    energy = 0.
    state = system.SystemState(pos, vel, alpha, energy)

    state = runner.minimize_then_run(state)
    state = runner.run(state)

    assert state


if __name__ == '__main__':
    main()
