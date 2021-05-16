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

    def test_expected_resname_mismatch_should_raise(self):
        with self.assertRaises(KeyError):
            # This residue is an alanine
            self.system.atom_index(1, "CA", expected_resname="CYS")

    def test_expected_resname_should_handle_n_term(self):
        self.assertEqual(
            self.system.atom_index(0, "CA", expected_resname="ALA", chainid=0, one_based=False),
            4,
        )

    def test_expected_resname_should_handle_c_term(self):
        self.assertEqual(
            self.system.atom_index(2, "CA", expected_resname="ALA", chainid=0, one_based=False),
            24,
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

class TestResidueIndexingExpectedResname(unittest.TestCase):
    def setUp(self):
        p1 = protein.SubSystemFromSequence("NALA CYS CLYS")
        p2 = protein.SubSystemFromSequence("NARG TRP CTYR")
        b = builder.SystemBuilder()
        self.system = b.build_system([p1, p2])

    def test_expected_resname_mismatch_should_raise(self):
        with self.assertRaises(KeyError):
            # This residue is CYS, not a VAL
            self.system.residue_index(1, expected_resname="VAL", one_based=False)

    def test_expected_resname_should_match(self):
        self.system.residue_index(0, expected_resname="ALA", one_based=False)
        self.system.residue_index(1, expected_resname="CYS", one_based=False)
        self.system.residue_index(2, expected_resname="LYS", one_based=False)
        self.system.residue_index(3, expected_resname="ARG", one_based=False)
        self.system.residue_index(4, expected_resname="TRP", one_based=False)
        self.system.residue_index(5, expected_resname="TYR", one_based=False)

class TestResidueIndexingHistidine(unittest.TestCase):
    def test_hid_should_match_his(self):
        p1 = protein.SubSystemFromSequence("NALA HID CLYS")
        b = builder.SystemBuilder()
        system = b.build_system([p1])
        system.residue_index(1, expected_resname="HIS", one_based=False)

    def test_hie_should_match_his(self):
        p1 = protein.SubSystemFromSequence("NALA HIE CLYS")
        b = builder.SystemBuilder()
        system = b.build_system([p1])
        system.residue_index(1, expected_resname="HIS", one_based=False)

    def test_hip_should_match_his(self):
        p1 = protein.SubSystemFromSequence("NALA HIP CLYS")
        b = builder.SystemBuilder()
        system = b.build_system([p1])
        system.residue_index(1, expected_resname="HIS", one_based=False)

class TestResidueIndexingAsparticAcid(unittest.TestCase):
    def test_ash_should_match_asp(self):
        p1 = protein.SubSystemFromSequence("NALA ASH CLYS")
        b = builder.SystemBuilder()
        system = b.build_system([p1])
        system.residue_index(1, expected_resname="ASP", one_based=False)

class TestResidueIndexingGlutamicAcid(unittest.TestCase):
    def test_ash_should_match_asp(self):
        p1 = protein.SubSystemFromSequence("NALA GLH CLYS")
        b = builder.SystemBuilder()
        system = b.build_system([p1])
        system.residue_index(1, expected_resname="GLU", one_based=False)

class TestResidueIndexingLysine(unittest.TestCase):
    def test_lyn_should_match_lys(self):
        p1 = protein.SubSystemFromSequence("NALA LYN CLYS")
        b = builder.SystemBuilder()
        system = b.build_system([p1])
        system.residue_index(1, expected_resname="LYS", one_based=False)

class TestResidueIndexingDisulfide(unittest.TestCase):
    def test_lyn_should_match_lys(self):
        p = protein.SubSystemFromSequence("NCYX ALA CCYX")
        p.add_disulfide(0, 2)
        b = builder.SystemBuilder()
        system = b.build_system([p])
        system.residue_index(0, expected_resname="CYS", one_based=False)
        system.residue_index(2, expected_resname="CYS", one_based=False)