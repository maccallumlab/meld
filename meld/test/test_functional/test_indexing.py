import unittest
from meld.system import (
    protein,
    builder,
)


class TestAtomIndexingOneBased(unittest.TestCase):
    def setUp(self):
        p1 = protein.ProteinMoleculeFromSequence("NALA ALA CALA")
        p2 = protein.ProteinMoleculeFromSequence("NALA ALA CALA")
        b = builder.SystemBuilder()
        self.system = b.build_system_from_molecules([p1, p2])

    def test_absolute_index(self):
        self.assertEqual(self.system.atom_index(1, "CA"), 4)
        self.assertEqual(self.system.atom_index(2, "CA"), 14)
        self.assertEqual(self.system.atom_index(3, "CA"), 24)
        self.assertEqual(self.system.atom_index(4, "CA"), 37)
        self.assertEqual(self.system.atom_index(5, "CA"), 47)
        self.assertEqual(self.system.atom_index(6, "CA"), 57)

    def test_relative_index(self):
        self.assertEqual(self.system.atom_index(1, "CA", chainid=1), 4)
        self.assertEqual(self.system.atom_index(2, "CA", chainid=1), 14)
        self.assertEqual(self.system.atom_index(3, "CA", chainid=1), 24)
        self.assertEqual(self.system.atom_index(1, "CA", chainid=2), 37)
        self.assertEqual(self.system.atom_index(2, "CA", chainid=2), 47)
        self.assertEqual(self.system.atom_index(3, "CA", chainid=2), 57)


class TestAtomIndexingZeroBased(unittest.TestCase):
    def setUp(self):
        p1 = protein.ProteinMoleculeFromSequence("NALA ALA CALA")
        p2 = protein.ProteinMoleculeFromSequence("NALA ALA CALA")
        b = builder.SystemBuilder()
        self.system = b.build_system_from_molecules([p1, p2])

    def test_absolute_index(self):
        self.assertEqual(self.system.atom_index(0, "CA", one_based=False), 4)
        self.assertEqual(self.system.atom_index(1, "CA", one_based=False), 14)
        self.assertEqual(self.system.atom_index(2, "CA", one_based=False), 24)
        self.assertEqual(self.system.atom_index(3, "CA", one_based=False), 37)
        self.assertEqual(self.system.atom_index(4, "CA", one_based=False), 47)
        self.assertEqual(self.system.atom_index(5, "CA", one_based=False), 57)

    def test_relative_index(self):
        self.assertEqual(self.system.atom_index(0, "CA", chainid=0, one_based=False), 4)
        self.assertEqual(
            self.system.atom_index(1, "CA", chainid=0, one_based=False), 14
        )
        self.assertEqual(
            self.system.atom_index(2, "CA", chainid=0, one_based=False), 24
        )
        self.assertEqual(
            self.system.atom_index(0, "CA", chainid=1, one_based=False), 37
        )
        self.assertEqual(
            self.system.atom_index(1, "CA", chainid=1, one_based=False), 47
        )
        self.assertEqual(
            self.system.atom_index(2, "CA", chainid=1, one_based=False), 57
        )


class TestResidueIndexingOneBased(unittest.TestCase):
    def setUp(self):
        p1 = protein.ProteinMoleculeFromSequence("NALA ALA CALA")
        p2 = protein.ProteinMoleculeFromSequence("NALA ALA CALA")
        b = builder.SystemBuilder()
        self.system = b.build_system_from_molecules([p1, p2])

    def test_absolute_index(self):
        self.assertEqual(self.system.residue_index(1), 0)
        self.assertEqual(self.system.residue_index(2), 1)
        self.assertEqual(self.system.residue_index(3), 2)
        self.assertEqual(self.system.residue_index(4), 3)
        self.assertEqual(self.system.residue_index(5), 4)
        self.assertEqual(self.system.residue_index(6), 5)

    def test_relative_index(self):
        self.assertEqual(self.system.residue_index(1, chainid=1), 0)
        self.assertEqual(self.system.residue_index(2, chainid=1), 1)
        self.assertEqual(self.system.residue_index(3, chainid=1), 2)
        self.assertEqual(self.system.residue_index(1, chainid=2), 3)
        self.assertEqual(self.system.residue_index(2, chainid=2), 4)
        self.assertEqual(self.system.residue_index(3, chainid=2), 5)


class TestResidueIndexingZeroBased(unittest.TestCase):
    def setUp(self):
        p1 = protein.ProteinMoleculeFromSequence("NALA ALA CALA")
        p2 = protein.ProteinMoleculeFromSequence("NALA ALA CALA")
        b = builder.SystemBuilder()
        self.system = b.build_system_from_molecules([p1, p2])

    def test_absolute_index(self):
        self.assertEqual(self.system.residue_index(0, one_based=False), 0)
        self.assertEqual(self.system.residue_index(1, one_based=False), 1)
        self.assertEqual(self.system.residue_index(2, one_based=False), 2)
        self.assertEqual(self.system.residue_index(3, one_based=False), 3)
        self.assertEqual(self.system.residue_index(4, one_based=False), 4)
        self.assertEqual(self.system.residue_index(5, one_based=False), 5)

    def test_relative_index(self):
        self.assertEqual(self.system.residue_index(0, chainid=0, one_based=False), 0)
        self.assertEqual(self.system.residue_index(1, chainid=0, one_based=False), 1)
        self.assertEqual(self.system.residue_index(2, chainid=0, one_based=False), 2)
        self.assertEqual(self.system.residue_index(0, chainid=1, one_based=False), 3)
        self.assertEqual(self.system.residue_index(1, chainid=1, one_based=False), 4)
        self.assertEqual(self.system.residue_index(2, chainid=1, one_based=False), 5)
