#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import unittest
import subprocess


class CommTestCase(unittest.TestCase):
    def test_broadcast_lambda(self):
        subprocess.check_call('mpirun -np 4 python broadcast_alpha.py', shell=True)

    def test_broadcast_states(self):
        subprocess.check_call('mpirun -np 4 python broadcast_states.py', shell=True)

    def test_broadcast_states_for_energy_calc(self):
        subprocess.check_call('mpirun -np 4 python broadcast_states_for_energy_calc.py', shell=True)

    def test_gather_energies(self):
        subprocess.check_call('mpirun -np 4 python gather_energies.py', shell=True)

    def test_gather_states(self):
        subprocess.check_call('mpirun -np 4 python gather_states.py', shell=True)
