#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import unittest
from meld import AmberSubSystemFromSequence, AmberSystemBuilder, AmberOptions
from meld.system.temperature import ConstantTemperatureScaler
from openmm import unit as u  # type: ignore


class TestCreateFromSequence(unittest.TestCase):
    def setUp(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        spec = b.build_system([p])
        self.system = spec.finalize()

    def test_has_correct_number_of_atoms(self):
        self.assertEqual(self.system.n_atoms, 33)

    def test_coordinates_have_correct_shape(self):
        self.assertEqual(self.system.template_coordinates.shape[0], 33)
        self.assertEqual(self.system.template_coordinates.shape[1], 3)

    def test_has_correct_atom_names(self):
        self.assertEqual(self.system.atom_names[0], "N")
        self.assertEqual(self.system.atom_names[-1], "OXT")
        self.assertEqual(
            len(self.system.atom_names), self.system.template_coordinates.shape[0]
        )

    def test_has_correct_residue_indices(self):
        self.assertEqual(self.system.residue_numbers[0], 0)
        self.assertEqual(self.system.residue_numbers[-1], 2)
        self.assertEqual(
            len(self.system.residue_numbers), self.system.template_coordinates.shape[0]
        )

    def test_has_correct_residue_names(self):
        self.assertEqual(self.system.residue_names[0], "ALA")
        self.assertEqual(self.system.residue_names[-1], "ALA")
        self.assertEqual(sum(["ALA" == res for res in self.system.residue_names]), 33)
        self.assertEqual(
            len(self.system.residue_names), self.system.template_coordinates.shape[0]
        )

    def test_temperature_scaler_defaults_to_none(self):
        self.assertEqual(self.system.temperature_scaler, None)


class TestCreateFromSequenceExplicit(TestCreateFromSequence):
    def setUp(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        options = AmberOptions(
            solvation="explicit",
            enable_pme=True,
            enable_pressure_coupling=True,
            enable_amap=False,
            cutoff=0.9,
        )
        b = AmberSystemBuilder(options)
        spec = b.build_system([p])
        self.system = spec.finalize()

    def test_has_correct_number_of_atoms(self):
        self.assertEqual(self.system.n_atoms, 861)

    def test_coordinates_have_correct_shape(self):
        self.assertEqual(self.system.template_coordinates.shape[0], 861)
        self.assertEqual(self.system.template_coordinates.shape[1], 3)

    def test_has_correct_atom_names(self):
        self.assertEqual(self.system.atom_names[0], "N")
        self.assertEqual(self.system.atom_names[-1], "H2")
        self.assertEqual(
            len(self.system.atom_names), self.system.template_coordinates.shape[0]
        )

    def test_has_correct_residue_indices(self):
        self.assertEqual(self.system.residue_numbers[0], 0)
        self.assertEqual(self.system.residue_numbers[-1], 278)
        self.assertEqual(
            len(self.system.residue_numbers), self.system.template_coordinates.shape[0]
        )

    def test_has_correct_residue_names(self):
        self.assertEqual(self.system.residue_names[0], "ALA")
        self.assertEqual(self.system.residue_names[-1], "HOH")
        self.assertEqual(sum(["ALA" == res for res in self.system.residue_names]), 33)
        self.assertEqual(sum(["HOH" == res for res in self.system.residue_names]), 828)
        self.assertEqual(
            len(self.system.residue_names), self.system.template_coordinates.shape[0]
        )


class TestCreateFromSequenceExplicitIons(TestCreateFromSequence):
    def setUp(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        options = AmberOptions(
            solvation="explicit",
            explicit_ions=True,
            p_ioncount=3,
            n_ioncount=3,
            enable_pme=True,
            enable_pressure_coupling=True,
            enable_amap=False,
            cutoff=0.9,
        )
        b = AmberSystemBuilder(options)
        spec = b.build_system([p])
        self.system = spec.finalize()

    def test_has_correct_number_of_atoms(self):
        self.assertEqual(self.system.n_atoms, 849)

    def test_coordinates_have_correct_shape(self):
        self.assertEqual(self.system.template_coordinates.shape[0], 849)
        self.assertEqual(self.system.template_coordinates.shape[1], 3)

    def test_has_correct_atom_names(self):
        self.assertEqual(self.system.atom_names[0], "N")
        self.assertEqual(self.system.atom_names[-1], "H2")
        self.assertEqual(
            len(self.system.atom_names), self.system.template_coordinates.shape[0]
        )

    def test_has_correct_residue_indices(self):
        self.assertEqual(self.system.residue_numbers[0], 0)
        self.assertEqual(self.system.residue_numbers[-1], 278)
        self.assertEqual(
            len(self.system.residue_numbers), self.system.template_coordinates.shape[0]
        )

    def test_has_correct_residue_names(self):
        self.assertEqual(self.system.residue_names[0], "ALA")
        self.assertEqual(self.system.residue_names[-1], "HOH")
        self.assertEqual(sum(["ALA" == res for res in self.system.residue_names]), 33)
        self.assertEqual(sum(["HOH" == res for res in self.system.residue_names]), 810)
        self.assertEqual(sum(["Na+" == res for res in self.system.residue_names]), 3)
        self.assertEqual(sum(["Cl-" == res for res in self.system.residue_names]), 3)
        self.assertEqual(
            len(self.system.residue_names), self.system.template_coordinates.shape[0]
        )


class TestCreateFromSequenceExplicitIonsNeutralize(TestCreateFromSequence):
    def setUp(self):
        p = AmberSubSystemFromSequence("NALA GLU CALA")
        options = AmberOptions(
            solvation="explicit",
            explicit_ions=True,
            enable_pme=True,
            enable_pressure_coupling=True,
            enable_amap=False,
            cutoff=0.9,
        )
        b = AmberSystemBuilder(options)
        spec = b.build_system([p])
        self.system = spec.finalize()

    def test_has_correct_number_of_atoms(self):
        self.assertEqual(self.system.n_atoms, 948)

    def test_coordinates_have_correct_shape(self):
        self.assertEqual(self.system.template_coordinates.shape[0], 948)
        self.assertEqual(self.system.template_coordinates.shape[1], 3)

    def test_has_correct_atom_names(self):
        self.assertEqual(self.system.atom_names[0], "N")
        self.assertEqual(self.system.atom_names[-1], "H2")
        self.assertEqual(
            len(self.system.atom_names), self.system.template_coordinates.shape[0]
        )

    def test_has_correct_residue_indices(self):
        self.assertEqual(self.system.residue_numbers[0], 0)
        self.assertEqual(self.system.residue_numbers[-1], 306)
        self.assertEqual(
            len(self.system.residue_numbers), self.system.template_coordinates.shape[0]
        )

    def test_has_correct_residue_names(self):
        self.assertEqual(self.system.residue_names[0], "ALA")
        self.assertEqual(self.system.residue_names[-1], "HOH")
        self.assertEqual(sum(["ALA" == res for res in self.system.residue_names]), 23)
        self.assertEqual(sum(["HOH" == res for res in self.system.residue_names]), 909)
        self.assertEqual(sum(["Na+" == res for res in self.system.residue_names]), 1)
        self.assertEqual(
            len(self.system.residue_names), self.system.template_coordinates.shape[0]
        )


class TestConstantTemperatureScaler(unittest.TestCase):
    def setUp(self):
        self.s = ConstantTemperatureScaler(300.0 * u.kelvin)

    def test_returns_constant_when_alpha_is_zero(self):
        t = self.s(0.0)
        self.assertAlmostEqual(t, 300.0)

    def test_returns_constant_when_alpha_is_one(self):
        t = self.s(1.0)
        self.assertAlmostEqual(t, 300.0)

    def test_raises_when_alpha_below_zero(self):
        with self.assertRaises(RuntimeError):
            self.s(-1)

    def test_raises_when_alpha_above_one(self):
        with self.assertRaises(RuntimeError):
            self.s(2)
