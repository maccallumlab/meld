#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import unittest
import openmm as mm  # type: ignore
from meld import AmberSubSystemFromSequence, AmberSystemBuilder, AmberOptions


def _find_cmap_force(system):
    for force in system.getForces():
        if isinstance(force, mm.CMAPTorsionForce):
            return force
    raise RuntimeError("CMAP force not found")


class TestAddAMAPTriAla(unittest.TestCase):
    def setUp(self):
        # create a tri-ala molecule
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        options = AmberOptions(enable_amap=True)
        b = AmberSystemBuilder(options)
        self.spec = b.build_system([p])

    def test_should_add_force_to_system(self):
        force_types = [type(f) for f in self.spec.system.getForces()]
        self.assertIn(mm.CMAPTorsionForce, force_types)

    def test_number_of_torsions_should_be_correct(self):
        EXPECTED = 1

        force = _find_cmap_force(self.spec.system)
        actual = force.getNumTorsions()

        self.assertEqual(actual, EXPECTED)

    def test_number_of_maps_should_be_correct(self):
        EXPECTED = 4

        force = _find_cmap_force(self.spec.system)
        actual = force.getNumMaps()

        self.assertEqual(actual, EXPECTED)

    def test_correct_torsion_should_be_added(self):
        EXPECTED = [2, 10, 12, 14, 20, 12, 14, 20, 22]

        force = _find_cmap_force(self.spec.system)
        actual = force.getTorsionParameters(0)

        self.assertEqual(actual, EXPECTED)


class TestAddAMAPDoubleTriAla(unittest.TestCase):
    def setUp(self):
        # create a tri-ala molecule
        p1 = AmberSubSystemFromSequence("NALA ALA CALA")
        p2 = AmberSubSystemFromSequence("NALA ALA CALA")
        options = AmberOptions(enable_amap=True)
        b = AmberSystemBuilder(options)
        self.spec = b.build_system([p1, p2])

    def test_should_add_force_to_system(self):
        force_types = [type(f) for f in self.spec.system.getForces()]
        self.assertIn(mm.CMAPTorsionForce, force_types)

    def test_number_of_torsions_should_be_correct(self):
        EXPECTED = 2

        force = _find_cmap_force(self.spec.system)
        actual = force.getNumTorsions()

        self.assertEqual(actual, EXPECTED)

    def test_number_of_maps_should_be_correct(self):
        EXPECTED = 4

        force = _find_cmap_force(self.spec.system)
        actual = force.getNumMaps()

        self.assertEqual(actual, EXPECTED)

    def test_correct_torsion_should_be_added(self):
        EXPECTED1 = [2, 10, 12, 14, 20, 12, 14, 20, 22]
        EXPECTED2 = [2, 43, 45, 47, 53, 45, 47, 53, 55]

        force = _find_cmap_force(self.spec.system)
        actual1 = force.getTorsionParameters(0)
        actual2 = force.getTorsionParameters(1)

        self.assertEqual(actual1, EXPECTED1)
        self.assertEqual(actual2, EXPECTED2)


class TestAddsCorrectMapType(unittest.TestCase):
    def test_Gly_has_correct_map(self):
        EXPECTED = 0

        spec = self.get_spec("NALA GLY CALA")
        force = _find_cmap_force(spec.system)
        actual = force.getTorsionParameters(0)[0]

        self.assertEqual(actual, EXPECTED)

    def test_Pro_has_correct_map(self):
        EXPECTED = 1

        spec = self.get_spec("NALA PRO CALA")
        force = _find_cmap_force(spec.system)
        actual = force.getTorsionParameters(0)[0]

        self.assertEqual(actual, EXPECTED)

    def test_Ala_has_correct_map(self):
        EXPECTED = 2

        spec = self.get_spec("NALA ALA CALA")
        force = _find_cmap_force(spec.system)
        actual = force.getTorsionParameters(0)[0]

        self.assertEqual(actual, EXPECTED)

    def test_Cys_has_correct_map(self):
        EXPECTED = 3

        spec = self.get_spec("NALA CYS CALA")
        force = _find_cmap_force(spec.system)
        actual = force.getTorsionParameters(0)[0]

        self.assertEqual(actual, EXPECTED)

    def test_CYX_has_correct_map(self):
        EXPECTED = 3

        spec = self.get_spec("NALA CYX CALA")
        force = _find_cmap_force(spec.system)
        actual = force.getTorsionParameters(0)[0]

        self.assertEqual(actual, EXPECTED)

    def test_TRP_has_correct_map(self):
        EXPECTED = 3

        spec = self.get_spec("NALA TRP CALA")
        force = _find_cmap_force(spec.system)
        actual = force.getTorsionParameters(0)[0]

        self.assertEqual(actual, EXPECTED)

    # We could test all of the amino acids here, but that would be
    # tedious. The selection above ought to be sufficient to
    # demonstrate that the algorithm is working.

    def get_spec(self, sequence):
        p = AmberSubSystemFromSequence(sequence)
        options = AmberOptions(enable_amap=True)
        b = AmberSystemBuilder(options)
        return b.build_system([p])
