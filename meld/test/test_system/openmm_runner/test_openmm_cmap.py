#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import unittest

from simtk import openmm
import mock
from mock import ANY
import numpy as np

from meld.system import protein, builder
from meld.system.openmm_runner.cmap import CMAPAdder


class TestAddCMAPTriAla(unittest.TestCase):
    def setUp(self):
        # create a tri-ala molecule
        p = protein.ProteinMoleculeFromSequence('NALA ALA CALA')
        b = builder.SystemBuilder()
        self.system = b.build_system_from_molecules([p])

        # create eight 24x24 maps filled with 0, 1, ..7
        self.maps = [np.zeros((24, 24)) + i for i in range(8)]

        # mock openmm system to recieve the new cmap torsion force
        self.mock_openmm_system = mock.Mock(spec=openmm.System)

        # patch out CMAPTorsionForce so we can see how it is called
        self.patcher = mock.patch('meld.system.openmm_runner.cmap.openmm.CMAPTorsionForce',
                                  spec=openmm.CMAPTorsionForce)
        self.MockCMAP = self.patcher.start()
        self.mock_cmap = mock.Mock(spec=openmm.CMAPTorsionForce)
        self.MockCMAP.return_value = self.mock_cmap

    def tearDown(self):
        self.patcher.stop()

    def test_adds_maps_to_force(self):
        with mock.patch('meld.system.openmm_runner.cmap.np.loadtxt') as mock_loadtxt:
            # this is a bit hacky and depends on what order the maps are loaded in
            expected_gly = 3.0 * 0. + 7.0 * 1. + np.zeros((24, 24)).flatten()
            expected_pro = 3.0 * 2. + 7.0 * 3. + np.zeros((24, 24)).flatten()
            expected_ala = 3.0 * 4. + 7.0 * 5. + np.zeros((24, 24)).flatten()
            expected_gen = 3.0 * 6. + 7.0 * 7. + np.zeros((24, 24)).flatten()
            mock_loadtxt.side_effect = self.maps
            adder = CMAPAdder(self.system.top_string, alpha_bias=3.0, beta_bias=7.0)

            adder.add_to_openmm(self.mock_openmm_system)

            self.assertEqual(self.MockCMAP.call_count, 1)
            self.assertEqual(self.mock_cmap.addMap.call_count, 4)

            # make sure all of the sizes are correct
            add_map_args = self.mock_cmap.addMap.call_args_list
            self.assertEqual(add_map_args[0][0][0], 24)
            self.assertEqual(add_map_args[1][0][0], 24)
            self.assertEqual(add_map_args[2][0][0], 24)
            self.assertEqual(add_map_args[3][0][0], 24)

            # make sure the maps are correct
            np.testing.assert_almost_equal(add_map_args[0][0][1], expected_gly)
            np.testing.assert_almost_equal(add_map_args[1][0][1], expected_pro)
            np.testing.assert_almost_equal(add_map_args[2][0][1], expected_ala)
            np.testing.assert_almost_equal(add_map_args[3][0][1], expected_gen)

    def test_correct_torsions_should_be_added_to_force(self):
        adder = CMAPAdder(self.system.top_string)
        adder.add_to_openmm(self.mock_openmm_system)

        # map should be #2 for alanine
        # all atom indices are zero-based in openmm
        self.mock_cmap.addTorsion.assert_called_once_with(2, 10, 12, 14, 20, 12, 14, 20, 22)

    def test_force_should_be_added_to_system(self):
        adder = CMAPAdder(self.system.top_string)
        adder.add_to_openmm(self.mock_openmm_system)

        self.mock_openmm_system.addForce.assert_called_once_with(self.mock_cmap)


class TestAddCMAPDoubleTriAla(unittest.TestCase):
    def setUp(self):
        # create a tri-ala molecule
        p = protein.ProteinMoleculeFromSequence('NALA ALA CALA')
        b = builder.SystemBuilder()
        self.system = b.build_system_from_molecules([p, p])

        # mock openmm system to recieve the new cmap torsion force
        self.mock_openmm_system = mock.Mock(spec=openmm.System)

        # patch out CMAPTorsionForce so we can see how it is called
        self.patcher = mock.patch('meld.system.openmm_runner.cmap.openmm.CMAPTorsionForce',
                                  spec=openmm.CMAPTorsionForce)
        self.MockCMAP = self.patcher.start()
        self.mock_cmap = mock.Mock(spec=openmm.CMAPTorsionForce)
        self.MockCMAP.return_value = self.mock_cmap

    def tearDown(self):
        self.patcher.stop()

    def test_correct_torsions_should_be_added_to_force(self):
        adder = CMAPAdder(self.system.top_string)
        adder.add_to_openmm(self.mock_openmm_system)

        # map should be #2 for alanine
        # all atom indices are zero-based in openmm
        expected_calls = [
            mock.call(2, 10, 12, 14, 20, 12, 14, 20, 22),
            mock.call(2, 43, 45, 47, 53, 45, 47, 53, 55)
        ]
        self.mock_cmap.addTorsion.assert_has_calls(expected_calls)


class TestAddsCorrectMapType(unittest.TestCase):
    def setUp(self):
        # mock openmm system to recieve the new cmap torsion force
        self.mock_openmm_system = mock.Mock(spec=openmm.System)

        # patch out CMAPTorsionForce so we can see how it is called
        self.patcher = mock.patch('meld.system.openmm_runner.cmap.openmm.CMAPTorsionForce',
                                  spec=openmm.CMAPTorsionForce)
        self.MockCMAP = self.patcher.start()
        self.mock_cmap = mock.Mock(spec=openmm.CMAPTorsionForce)
        self.MockCMAP.return_value = self.mock_cmap

    def tearDown(self):
        self.patcher.stop()

    def make_system(self, restype):
        sequence = 'NALA {0} CALA'.format(restype)
        p = protein.ProteinMoleculeFromSequence(sequence)
        b = builder.SystemBuilder()
        self.system = b.build_system_from_molecules([p])

    def test_correct_map_used_for_GLY(self):
        self.make_system('GLY')

        adder = CMAPAdder(self.system.top_string)
        adder.add_to_openmm(self.mock_openmm_system)

        self.mock_cmap.addTorsion.assert_called_once_with(0, ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY)

    def test_correct_map_used_for_PRO(self):
        self.make_system('PRO')

        adder = CMAPAdder(self.system.top_string)
        adder.add_to_openmm(self.mock_openmm_system)

        self.mock_cmap.addTorsion.assert_called_once_with(1, ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY)

    def test_correct_map_used_for_ALA(self):
        self.make_system('ALA')

        adder = CMAPAdder(self.system.top_string)
        adder.add_to_openmm(self.mock_openmm_system)

        self.mock_cmap.addTorsion.assert_called_once_with(2, ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY)

    def test_correct_map_used_for_general_case(self):
        res_names = ['CYS', 'CYX', 'ASP', 'ASH', 'GLU', 'GLH', 'PHE', 'HIS', 'HID', 'HIE', 'HIP',
                     'LYS', 'LYN', 'MET', 'SER', 'TRP', 'TYR', 'ILE', 'ASN', 'GLN', 'THR', 'TRP',
                     'LEU', 'ARG']
        for res in res_names:
            self.mock_cmap.reset_mock()
            self.make_system(res)

            adder = CMAPAdder(self.system.top_string)
            adder.add_to_openmm(self.mock_openmm_system)

            self.mock_cmap.addTorsion.assert_called_once_with(3, ANY, ANY, ANY, ANY, ANY, ANY, ANY, ANY)
