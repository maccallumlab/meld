#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import unittest
import os
import subprocess


class CommTestCase(unittest.TestCase):
    def get_directory(self):
        directory = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../../")
        )
        return directory

    @unittest.skip("Problems with mpirun in tests")
    def test_broadcast_alpha(self):
        print("Started broadcast_alpha")
        directory = self.get_directory()
        path = os.path.join(directory, 'meld/test/test_functional/test_comm/broadcast_alpha.py')
        subprocess.check_call(f"PYTHONPATH={directory} mpirun -np 4 python {path}", shell=True)

    @unittest.skip("Problems with mpirun in tests")
    def test_broadcast_states(self):
        directory = self.get_directory()
        path = os.path.join(directory, 'meld/test/test_functional/test_comm/broadcast_states.py')
        subprocess.check_call(f"PYTHONPATH={directory} mpirun -np 4 python {path}", shell=True)

    @unittest.skip("Problems with mpirun in tests")
    def test_broadcast_states_for_energy_calc(self):
        directory = self.get_directory()
        path = os.path.join(directory, 'meld/test/test_functional/test_comm/broadcast_states_for_energy_calc.py')
        subprocess.check_call(f"PYTHONPATH={directory} mpirun -np 4 python {path}", shell=True)

    @unittest.skip("Problems with mpirun in tests")
    def test_gather_energies(self):
        directory = self.get_directory()
        path = os.path.join(directory, 'meld/test/test_functional/test_comm/gather_energies.py')
        subprocess.check_call(f"PYTHONPATH={directory} mpirun -np 4 python {path}", shell=True)

    @unittest.skip("Problems with mpirun in tests")
    def test_gather_states(self):
        directory = self.get_directory()
        path = os.path.join(directory, 'meld/test/test_functional/test_comm/gather_states.py')
        subprocess.check_call(f"PYTHONPATH={directory} mpirun -np 4 python {path}", shell=True)
