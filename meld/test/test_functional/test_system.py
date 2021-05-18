#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import unittest
from meld.system.subsystem import SubSystemFromSequence, SubSystemFromPdbFile
from meld.system.builder import SystemBuilder
from meld.system.temperature import ConstantTemperatureScaler, LinearTemperatureScaler, GeometricTemperatureScaler
from meld.system.options import RunOptions
from meld.system.meld_system import ParmTopReader


class TestCreateFromSequence(unittest.TestCase):
    def setUp(self):
        p = SubSystemFromSequence("NALA ALA CALA")
        b = SystemBuilder()
        self.system = b.build_system([p])

    def test_has_correct_number_of_atoms(self):
        self.assertEqual(self.system.n_atoms, 33)

    def test_coordinates_have_correct_shape(self):
        self.assertEqual(self.system.coordinates.shape[0], 33)
        self.assertEqual(self.system.coordinates.shape[1], 3)

    def test_has_correct_atom_names(self):
        self.assertEqual(self.system.atom_names[0], "N")
        self.assertEqual(self.system.atom_names[-1], "OXT")
        self.assertEqual(len(self.system.atom_names), self.system.coordinates.shape[0])

    def test_has_correct_residue_indices(self):
        self.assertEqual(self.system.residue_numbers[0], 1)
        self.assertEqual(self.system.residue_numbers[-1], 3)
        self.assertEqual(
            len(self.system.residue_numbers), self.system.coordinates.shape[0]
        )

    def test_has_correct_residue_names(self):
        self.assertEqual(self.system.residue_names[0], "ALA")
        self.assertEqual(self.system.residue_names[-1], "ALA")
        self.assertEqual(sum(["ALA" == res for res in self.system.residue_names]), 33)
        self.assertEqual(
            len(self.system.residue_names), self.system.coordinates.shape[0]
        )

    def test_temperature_scaler_defaults_to_none(self):
        self.assertEqual(self.system.temperature_scaler, None)


class TestCreateFromSequenceExplicit(TestCreateFromSequence):
    def setUp(self):
        p = SubSystemFromSequence("NALA ALA CALA")
        b = SystemBuilder(explicit_solvent=True)
        self.system = b.build_system([p])

    def test_has_correct_number_of_atoms(self):
        self.assertEqual(self.system.n_atoms, 861)

    def test_coordinates_have_correct_shape(self):
        self.assertEqual(self.system.coordinates.shape[0], 861)
        self.assertEqual(self.system.coordinates.shape[1], 3)

    def test_has_correct_atom_names(self):
        self.assertEqual(self.system.atom_names[0], "N")
        self.assertEqual(self.system.atom_names[-1], "H2")
        self.assertEqual(len(self.system.atom_names), self.system.coordinates.shape[0])

    def test_has_correct_residue_indices(self):
        self.assertEqual(self.system.residue_numbers[0], 1)
        self.assertEqual(self.system.residue_numbers[-1], 279)
        self.assertEqual(
            len(self.system.residue_numbers), self.system.coordinates.shape[0]
        )

    def test_has_correct_residue_names(self):
        self.assertEqual(self.system.residue_names[0], "ALA")
        self.assertEqual(self.system.residue_names[-1], "WAT")
        self.assertEqual(sum(["ALA" == res for res in self.system.residue_names]), 33)
        self.assertEqual(sum(["WAT" == res for res in self.system.residue_names]), 828)
        self.assertEqual(
            len(self.system.residue_names), self.system.coordinates.shape[0]
        )


class TestCreateFromSequenceExplicitIons(TestCreateFromSequence):
    def setUp(self):
        p = SubSystemFromSequence("NALA ALA CALA")
        b = SystemBuilder(
            explicit_solvent=True, explicit_ions=True, p_ioncount=3, n_ioncount=3
        )
        self.system = b.build_system([p])

    def test_has_correct_number_of_atoms(self):
        self.assertEqual(self.system.n_atoms, 849)

    def test_coordinates_have_correct_shape(self):
        self.assertEqual(self.system.coordinates.shape[0], 849)
        self.assertEqual(self.system.coordinates.shape[1], 3)

    def test_has_correct_atom_names(self):
        self.assertEqual(self.system.atom_names[0], "N")
        self.assertEqual(self.system.atom_names[-1], "H2")
        self.assertEqual(len(self.system.atom_names), self.system.coordinates.shape[0])

    def test_has_correct_residue_indices(self):
        self.assertEqual(self.system.residue_numbers[0], 1)
        self.assertEqual(self.system.residue_numbers[-1], 279)
        self.assertEqual(
            len(self.system.residue_numbers), self.system.coordinates.shape[0]
        )

    def test_has_correct_residue_names(self):
        self.assertEqual(self.system.residue_names[0], "ALA")
        self.assertEqual(self.system.residue_names[-1], "WAT")
        self.assertEqual(sum(["ALA" == res for res in self.system.residue_names]), 33)
        self.assertEqual(sum(["WAT" == res for res in self.system.residue_names]), 810)
        self.assertEqual(sum(["Na+" == res for res in self.system.residue_names]), 3)
        self.assertEqual(sum(["Cl-" == res for res in self.system.residue_names]), 3)
        self.assertEqual(
            len(self.system.residue_names), self.system.coordinates.shape[0]
        )


class TestCreateFromSequenceExplicitIonsNeutralize(TestCreateFromSequence):
    def setUp(self):
        p = SubSystemFromSequence("NALA GLU CALA")
        b = SystemBuilder(explicit_solvent=True, explicit_ions=True)
        self.system = b.build_system([p])

    def test_has_correct_number_of_atoms(self):
        self.assertEqual(self.system.n_atoms, 948)

    def test_coordinates_have_correct_shape(self):
        self.assertEqual(self.system.coordinates.shape[0], 948)
        self.assertEqual(self.system.coordinates.shape[1], 3)

    def test_has_correct_atom_names(self):
        self.assertEqual(self.system.atom_names[0], "N")
        self.assertEqual(self.system.atom_names[-1], "H2")
        self.assertEqual(len(self.system.atom_names), self.system.coordinates.shape[0])

    def test_has_correct_residue_indices(self):
        self.assertEqual(self.system.residue_numbers[0], 1)
        self.assertEqual(self.system.residue_numbers[-1], 307)
        self.assertEqual(
            len(self.system.residue_numbers), self.system.coordinates.shape[0]
        )

    def test_has_correct_residue_names(self):
        self.assertEqual(self.system.residue_names[0], "ALA")
        self.assertEqual(self.system.residue_names[-1], "WAT")
        self.assertEqual(sum(["ALA" == res for res in self.system.residue_names]), 23)
        self.assertEqual(sum(["WAT" == res for res in self.system.residue_names]), 909)
        self.assertEqual(sum(["Na+" == res for res in self.system.residue_names]), 1)
        self.assertEqual(
            len(self.system.residue_names), self.system.coordinates.shape[0]
        )


class TestConstantTemperatureScaler(unittest.TestCase):
    def setUp(self):
        self.s = ConstantTemperatureScaler(300.0)

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


class TestLinearTemperatureScaler(unittest.TestCase):
    def setUp(self):
        self.s = LinearTemperatureScaler(0.2, 0.8, 300.0, 500.0)

    def test_returns_min_when_alpha_is_low(self):
        t = self.s(0)
        self.assertAlmostEqual(t, 300.0)

    def test_returns_max_when_alpha_is_high(self):
        t = self.s(1)
        self.assertAlmostEqual(t, 500.0)

    def test_returns_mid_when_alpha_is_half(self):
        t = self.s(0.5)
        self.assertAlmostEqual(t, 400.0)

    def test_raises_when_alpha_below_zero(self):
        with self.assertRaises(RuntimeError):
            self.s(-1)

    def test_raises_when_alpha_above_one(self):
        with self.assertRaises(RuntimeError):
            self.s(2)

    def test_raises_when_alpha_min_below_zero(self):
        with self.assertRaises(RuntimeError):
            LinearTemperatureScaler(-0.1, 0.8, 300.0, 500.0)

    def test_raises_when_alpha_min_above_one(self):
        with self.assertRaises(RuntimeError):
            LinearTemperatureScaler(1.1, 0.8, 300.0, 500.0)

    def test_raises_when_alpha_max_below_zero(self):
        with self.assertRaises(RuntimeError):
            LinearTemperatureScaler(0.0, -0.1, 300.0, 500.0)

    def test_raises_when_alpha_max_above_one(self):
        with self.assertRaises(RuntimeError):
            LinearTemperatureScaler(0.0, 1.1, 300.0, 500.0)

    def test_raises_when_alpha_min_above_alpha_max(self):
        with self.assertRaises(RuntimeError):
            LinearTemperatureScaler(1.0, 0.0, 300.0, 500.0)

    def test_raises_when_temp_min_is_below_zero(self):
        with self.assertRaises(RuntimeError):
            LinearTemperatureScaler(0.0, 1.0, -300.0, 500.0)

    def test_raises_when_temp_max_is_below_zero(self):
        with self.assertRaises(RuntimeError):
            LinearTemperatureScaler(0.0, 1.0, 300.0, -500.0)


class TestGeometricTemperatureScaler(unittest.TestCase):
    def setUp(self):
        self.s = GeometricTemperatureScaler(0.2, 0.8, 300.0, 500.0)

    def test_returns_min_when_alpha_is_low(self):
        t = self.s(0)
        self.assertAlmostEqual(t, 300.0)

    def test_returns_max_when_alpha_is_high(self):
        t = self.s(1)
        self.assertAlmostEqual(t, 500.0)

    def test_returns_mid_when_alpha_is_half(self):
        t = self.s(0.5)
        self.assertAlmostEqual(t, 387.298334621)

    def test_raises_when_alpha_below_zero(self):
        with self.assertRaises(RuntimeError):
            self.s(-1)

    def test_raises_when_alpha_above_one(self):
        with self.assertRaises(RuntimeError):
            self.s(2)

    def test_raises_when_alpha_min_below_zero(self):
        with self.assertRaises(RuntimeError):
            GeometricTemperatureScaler(-0.1, 0.8, 300.0, 500.0)

    def test_raises_when_alpha_min_above_one(self):
        with self.assertRaises(RuntimeError):
            GeometricTemperatureScaler(1.1, 0.8, 300.0, 500.0)

    def test_raises_when_alpha_max_below_zero(self):
        with self.assertRaises(RuntimeError):
            GeometricTemperatureScaler(0.0, -0.1, 300.0, 500.0)

    def test_raises_when_alpha_max_above_one(self):
        with self.assertRaises(RuntimeError):
            GeometricTemperatureScaler(0.0, 1.1, 300.0, 500.0)

    def test_raises_when_alpha_min_above_alpha_max(self):
        with self.assertRaises(RuntimeError):
            GeometricTemperatureScaler(1.0, 0.0, 300.0, 500.0)

    def test_raises_when_temp_min_is_below_zero(self):
        with self.assertRaises(RuntimeError):
            GeometricTemperatureScaler(0.0, 1.0, -300.0, 500.0)

    def test_raises_when_temp_max_is_below_zero(self):
        with self.assertRaises(RuntimeError):
            GeometricTemperatureScaler(0.0, 1.0, 300.0, -500.0)


class TestOptions(unittest.TestCase):
    def setUp(self):
        self.options = RunOptions()

    def test_timesteps_initializes_to_5000(self):
        self.assertEqual(self.options.timesteps, 5000)

    def test_timesteps_below_1_raises(self):
        with self.assertRaises(RuntimeError):
            self.options.timesteps = 0

    def test_minimization_steps_initializes_to_1000(self):
        self.assertEqual(self.options.minimize_steps, 1000)

    def test_minimizesteps_below_1_raises(self):
        with self.assertRaises(RuntimeError):
            self.options.minimize_steps = 0

    def test_implicit_solvent_model_defaults_to_gbneck2(self):
        self.assertEqual(self.options.implicit_solvent_model, "gbNeck2")

    def test_implicit_solvent_model_accepts_gbneck(self):
        self.options.implicit_solvent_model = "gbNeck"
        self.assertEqual(self.options.implicit_solvent_model, "gbNeck")

    def test_implicit_solvent_model_accepts_obc(self):
        self.options.implicit_solvent_model = "obc"
        self.assertEqual(self.options.implicit_solvent_model, "obc")

    def test_bad_implicit_model_raises(self):
        with self.assertRaises(RuntimeError):
            self.options.implicit_solvent_model = "bad"

    def test_cutoff_defaults_to_none(self):
        self.assertIs(self.options.cutoff, None)

    def test_cutoff_accepts_positive_float(self):
        self.options.cutoff = 1.0
        self.assertEqual(self.options.cutoff, 1.0)

    def test_cutoff_should_raise_with_non_positive_value(self):
        with self.assertRaises(RuntimeError):
            self.options.cutoff = 0.0

    def test_use_big_timestep_defaults_to_false(self):
        self.assertEqual(self.options.use_big_timestep, False)

    def test_use_amap_defaults_to_false(self):
        self.assertEqual(self.options.use_amap, False)


class TestOptionsImplicitSanityCheck(unittest.TestCase):
    def setUp(self):
        self.options = RunOptions(solvation="implicit")

    def test_setting_enable_pme_should_raise(self):
        with self.assertRaises(ValueError):
            self.options.enable_pme = True

    def test_setting_enable_pressure_coupling_should_raise(self):
        with self.assertRaises(ValueError):
            self.options.enable_pressure_coupling = True


class TestOptionsExplicitSanityCheck(unittest.TestCase):
    def setUp(self):
        self.options = RunOptions(solvation="explicit")

    def test_setting_usse_amap_should_riase(self):
        with self.assertRaises(ValueError):
            self.options.use_amap = True


class TestGetBonds(unittest.TestCase):
    def setUp(self):
        p = SubSystemFromSequence("NALA ALA CALA")
        b = SystemBuilder()
        sys = b.build_system([p])
        self.bonds = ParmTopReader(sys.top_string).get_bonds()

    def test_correct_bonds_should_be_present(self):
        self.assertIn((0, 4), self.bonds)  # N -CA
        self.assertIn((4, 10), self.bonds)  # CA-C
        self.assertIn((10, 12), self.bonds)  # C -N+1
        self.assertIn((12, 10), self.bonds)  # make sure we can find either order

    def test_wrong_bonds_should_not_be_present(self):
        self.assertNotIn((0, 10), self.bonds)  # N-C should not be present
