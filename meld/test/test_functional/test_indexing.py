import unittest
from meld.system import (
    protein,
    builder,
)


class TestAtomIndexingOneBased(unittest.TestCase):
    def setUp(self):
        p1 = protein.SubSystemFromSequence("NALA ALA CALA")
        p2 = protein.SubSystemFromSequence("NALA ALA CALA")
        b = builder.SystemBuilder()
        self.system = b.build_system([p1, p2])

    def test_absolute_index(self):
        self.assertEqual(self.system.atom_index(1, "CA", one_based=True), 4)
        self.assertEqual(self.system.atom_index(2, "CA", one_based=True), 14)
        self.assertEqual(self.system.atom_index(3, "CA", one_based=True), 24)
        self.assertEqual(self.system.atom_index(4, "CA", one_based=True), 37)
        self.assertEqual(self.system.atom_index(5, "CA", one_based=True), 47)
        self.assertEqual(self.system.atom_index(6, "CA", one_based=True), 57)

    def test_relative_index(self):
        self.assertEqual(self.system.atom_index(1, "CA", chainid=1, one_based=True), 4)
        self.assertEqual(self.system.atom_index(2, "CA", chainid=1, one_based=True), 14)
        self.assertEqual(self.system.atom_index(3, "CA", chainid=1, one_based=True), 24)
        self.assertEqual(self.system.atom_index(1, "CA", chainid=2, one_based=True), 37)
        self.assertEqual(self.system.atom_index(2, "CA", chainid=2, one_based=True), 47)
        self.assertEqual(self.system.atom_index(3, "CA", chainid=2, one_based=True), 57)


class TestAtomIndexingZeroBased(unittest.TestCase):
    def setUp(self):
        p1 = protein.SubSystemFromSequence("NALA ALA CALA")
        p2 = protein.SubSystemFromSequence("NALA ALA CALA")
        b = builder.SystemBuilder()
        self.system = b.build_system([p1, p2])

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
        p1 = protein.SubSystemFromSequence("NALA ALA CALA")
        p2 = protein.SubSystemFromSequence("NALA ALA CALA")
        b = builder.SystemBuilder()
        self.system = b.build_system([p1, p2])

    def test_absolute_index(self):
        self.assertEqual(self.system.residue_index(1, one_based=True), 0)
        self.assertEqual(self.system.residue_index(2, one_based=True), 1)
        self.assertEqual(self.system.residue_index(3, one_based=True), 2)
        self.assertEqual(self.system.residue_index(4, one_based=True), 3)
        self.assertEqual(self.system.residue_index(5, one_based=True), 4)
        self.assertEqual(self.system.residue_index(6, one_based=True), 5)

    def test_relative_index(self):
        self.assertEqual(self.system.residue_index(1, chainid=1, one_based=True), 0)
        self.assertEqual(self.system.residue_index(2, chainid=1, one_based=True), 1)
        self.assertEqual(self.system.residue_index(3, chainid=1, one_based=True), 2)
        self.assertEqual(self.system.residue_index(1, chainid=2, one_based=True), 3)
        self.assertEqual(self.system.residue_index(2, chainid=2, one_based=True), 4)
        self.assertEqual(self.system.residue_index(3, chainid=2, one_based=True), 5)


class TestResidueIndexingZeroBased(unittest.TestCase):
    def setUp(self):
        p1 = protein.SubSystemFromSequence("NALA ALA CALA")
        p2 = protein.SubSystemFromSequence("NALA ALA CALA")
        b = builder.SystemBuilder()
        self.system = b.build_system([p1, p2])

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
