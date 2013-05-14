import unittest
from meld.system import protein, builder, ConstantTemperatureScaler


class TestCreateFromSequence(unittest.TestCase):
    def setUp(self):
        p = protein.ProteinMoleculeFromSequence('NALA ALA CALA')
        b = builder.SystemBuilder()
        self.system = b.build_system_from_molecules([p])

    def test_has_correct_number_of_atoms(self):
        self.assertEqual(self.system.n_atoms, 33)

    def test_coordinates_have_correct_shape(self):
        self.assertEqual(self.system.coordinates.shape[0], 33)
        self.assertEqual(self.system.coordinates.shape[1], 3)

    def test_has_correct_atom_names(self):
        self.assertEqual(self.system.atom_names[0], 'N')
        self.assertEqual(self.system.atom_names[-1], 'OXT')
        self.assertEqual(len(self.system.atom_names), self.system.coordinates.shape[0])

    def test_has_correct_residue_indices(self):
        self.assertEqual(self.system.residue_numbers[0], 1)
        self.assertEqual(self.system.residue_numbers[-1], 3)
        self.assertEqual(len(self.system.residue_numbers), self.system.coordinates.shape[0])

    def test_has_correct_residue_names(self):
        self.assertEqual(self.system.residue_names[0], 'ALA')
        self.assertEqual(self.system.residue_names[-1], 'ALA')
        self.assertEqual(len(self.system.residue_names), self.system.coordinates.shape[0])

    def test_index_works(self):
        self.assertEqual(self.system.index_of_atom(1, 'N'), 1)
        self.assertEqual(self.system.index_of_atom(3, 'OXT'), 33)

    def test_temperature_scaler_defaults_to_none(self):
        self.assertEqual(self.system.temperature_scaler, None)


class TestConstantTemperatureScaler(unittest.TestCase):
    def setUp(self):
        self.s = ConstantTemperatureScaler(300.)

    def test_returns_constant_when_alpha_is_zero(self):
        t = self.s(0.)
        self.assertAlmostEqual(t, 300.)

    def test_returns_constant_when_alpha_is_one(self):
        t = self.s(1.)
        self.assertAlmostEqual(t, 300.)

    def test_raises_when_alpha_below_zero(self):
        with self.assertRaises(RuntimeError):
            self.s(-1)

    def test_raises_when_alpha_above_one(self):
        with self.assertRaises(RuntimeError):
            self.s(2)
