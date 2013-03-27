import unittest
from meld.system import protein, builder


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
