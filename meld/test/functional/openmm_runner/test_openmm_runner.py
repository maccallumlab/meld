from meld import system
import numpy
import unittest


class TestOpenRunner(unittest.TestCase):
    def test_runner(self):
        p = system.ProteinMoleculeFromSequence('NALA ALA CALA')
        b = system.SystemBuilder()
        sys = b.build_system_from_molecules([p])
        sys.temperature_scaler = system.ConstantTemperatureScaler(300.)

        options = system.RunOptions()
        options.timesteps = 10000

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
