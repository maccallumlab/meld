import unittest
from meld import (
    AmberSubSystemFromPdbFile,
    AmberSubSystemFromSequence,
    AmberSystemBuilder,
    AmberOptions,
)
from meld.system.indexing import ResidueIndex
from meld.util import in_temp_dir

pdb_string = """ATOM      1  N   ALA A   1       3.326   1.548  -0.000
ATOM      2  H1  ALA A   1       4.046   0.840  -0.000
ATOM      3  H2  ALA A   1       2.823   1.500  -0.875
ATOM      4  H3  ALA A   1       2.823   1.500   0.875
ATOM      5  CA  ALA A   1       3.970   2.846  -0.000
ATOM      6  HA  ALA A   1       3.672   3.400  -0.890
ATOM      7  CB  ALA A   1       3.577   3.654   1.232
ATOM      8  HB1 ALA A   1       3.877   3.116   2.131
ATOM      9  HB2 ALA A   1       4.075   4.623   1.206
ATOM     10  HB3 ALA A   1       2.497   3.801   1.241
ATOM     11  C   ALA A   1       5.486   2.705  -0.000
ATOM     12  O   ALA A   1       6.009   1.593  -0.000
ATOM     13  N   CYS A   2       6.191   3.839  -0.000
ATOM     14  H   CYS A   2       5.715   4.730  -0.000
ATOM     15  CA  CYS A   2       7.640   3.839  -0.000
ATOM     16  HA  CYS A   2       8.004   3.325   0.890
ATOM     17  CB  CYS A   2       8.189   3.127  -1.232
ATOM     18  HB2 CYS A   2       7.841   2.094  -1.241
ATOM     19  HB3 CYS A   2       7.841   3.636  -2.131
ATOM     20  SG  CYS A   2       9.992   3.050  -1.366
ATOM     21  HG  CYS A   2      10.018   2.385  -2.518
ATOM     22  C   CYS A   2       8.188   5.259  -0.000
ATOM     23  O   CYS A   2       7.425   6.222   0.000
ATOM     24  N   ASP A   3       9.517   5.386  -0.000
ATOM     25  H   ASP A   3      10.103   4.564  -0.000
ATOM     26  CA  ASP A   3      10.161   6.684   0.000
ATOM     27  HA  ASP A   3       9.863   7.239  -0.890
ATOM     28  CB  ASP A   3       9.768   7.492   1.232
ATOM     29  HB2 ASP A   3       8.688   7.640   1.241
ATOM     30  HB3 ASP A   3      10.068   6.954   2.131
ATOM     31  CG  ASP A   3      10.466   8.850   1.195
ATOM     32  OD1 ASP A   3       9.860   9.794   0.620
ATOM     33  OD2 ASP A   3      11.599   8.930   1.741
ATOM     34  C   ASP A   3      11.677   6.544  -0.000
ATOM     35  O   ASP A   3      12.200   5.432  -0.000
ATOM     36  OXT ASP A   3      12.395   7.541  -0.000
TER
ATOM     37  N   ARG B   4       3.326   1.548  -0.000
ATOM     38  H1  ARG B   4       4.046   0.840  -0.000
ATOM     39  H2  ARG B   4       2.823   1.500  -0.875
ATOM     40  H3  ARG B   4       2.823   1.500   0.875
ATOM     41  CA  ARG B   4       3.970   2.846  -0.000
ATOM     42  HA  ARG B   4       3.672   3.400  -0.890
ATOM     43  CB  ARG B   4       3.577   3.654   1.232
ATOM     44  HB2 ARG B   4       2.497   3.801   1.241
ATOM     45  HB3 ARG B   4       3.877   3.116   2.131
ATOM     46  CG  ARG B   4       4.274   5.010   1.195
ATOM     47  HG2 ARG B   4       5.354   4.863   1.186
ATOM     48  HG3 ARG B   4       3.974   5.548   0.296
ATOM     49  CD  ARG B   4       3.881   5.818   2.427
ATOM     50  HD2 ARG B   4       2.801   5.965   2.436
ATOM     51  HD3 ARG B   4       4.182   5.280   3.326
ATOM     52  NE  ARG B   4       4.540   7.143   2.424
ATOM     53  HE  ARG B   4       5.152   7.375   1.655
ATOM     54  CZ  ARG B   4       4.364   8.041   3.389
ATOM     55  NH1 ARG B   4       3.575   7.808   4.434
ATOM     56 HH11 ARG B   4       3.089   6.925   4.509
ATOM     57 HH12 ARG B   4       3.465   8.514   5.148
ATOM     58  NH2 ARG B   4       5.006   9.201   3.287
ATOM     59 HH21 ARG B   4       5.605   9.375   2.492
ATOM     60 HH22 ARG B   4       4.892   9.903   4.004
ATOM     61  C   ARG B   4       5.486   2.705  -0.000
ATOM     62  O   ARG B   4       6.009   1.593  -0.000
ATOM     63  N   LYS B   5       6.191   3.839  -0.000
ATOM     64  H   LYS B   5       5.715   4.730  -0.000
ATOM     65  CA  LYS B   5       7.640   3.839  -0.000
ATOM     66  HA  LYS B   5       8.004   3.325   0.890
ATOM     67  CB  LYS B   5       8.189   3.127  -1.232
ATOM     68  HB2 LYS B   5       7.841   2.094  -1.241
ATOM     69  HB3 LYS B   5       7.841   3.636  -2.131
ATOM     70  CG  LYS B   5       9.713   3.149  -1.195
ATOM     71  HG2 LYS B   5      10.062   4.181  -1.186
ATOM     72  HG3 LYS B   5      10.062   2.640  -0.296
ATOM     73  CD  LYS B   5      10.262   2.438  -2.427
ATOM     74  HD2 LYS B   5       9.914   1.405  -2.436
ATOM     75  HD3 LYS B   5       9.914   2.946  -3.326
ATOM     76  CE  LYS B   5      11.787   2.459  -2.389
ATOM     77  HE2 LYS B   5      12.136   3.492  -2.380
ATOM     78  HE3 LYS B   5      12.136   1.951  -1.491
ATOM     79  NZ  LYS B   5      12.316   1.773  -3.577
ATOM     80  HZ1 LYS B   5      11.993   2.245  -4.410
ATOM     81  HZ2 LYS B   5      13.326   1.788  -3.552
ATOM     82  HZ3 LYS B   5      11.993   0.817  -3.585
ATOM     83  C   LYS B   5       8.188   5.259  -0.000
ATOM     84  O   LYS B   5       7.425   6.222   0.000
ATOM     85  N   TRP B   6       9.517   5.386  -0.000
ATOM     86  H   TRP B   6      10.103   4.564  -0.000
ATOM     87  CA  TRP B   6      10.161   6.684   0.000
ATOM     88  HA  TRP B   6       9.863   7.239  -0.890
ATOM     89  CB  TRP B   6       9.768   7.492   1.232
ATOM     90  HB2 TRP B   6       8.688   7.640   1.241
ATOM     91  HB3 TRP B   6      10.068   6.954   2.131
ATOM     92  CG  TRP B   6      10.392   8.865   1.321
ATOM     93  CD1 TRP B   6      10.214   9.770   2.293
ATOM     94  HD1 TRP B   6       9.560   9.544   3.135
ATOM     95  NE1 TRP B   6      11.003  10.912   1.950
ATOM     96  HE1 TRP B   6      11.074  11.761   2.493
ATOM     97  CE2 TRP B   6      11.618  10.681   0.817
ATOM     98  CZ2 TRP B   6      12.488  11.528   0.120
ATOM     99  HZ2 TRP B   6      12.722  12.515   0.517
ATOM    100  CH2 TRP B   6      13.005  11.026  -1.069
ATOM    101  HH2 TRP B   6      13.689  11.630  -1.664
ATOM    102  CZ3 TRP B   6      12.674   9.792  -1.505
ATOM    103  HZ3 TRP B   6      13.089   9.414  -2.440
ATOM    104  CE3 TRP B   6      11.795   8.956  -0.786
ATOM    105  HE3 TRP B   6      11.550   7.965  -1.168
ATOM    106  CD2 TRP B   6      11.274   9.462   0.412
ATOM    107  C   TRP B   6      11.677   6.544  -0.000
ATOM    108  O   TRP B   6      12.200   5.432  -0.000
ATOM    109  OXT TRP B   6      12.395   7.541  -0.000
TER
ATOM    110  N   TYR     7       3.326   1.548  -0.000
ATOM    111  H1  TYR     7       4.046   0.840  -0.000
ATOM    112  H2  TYR     7       2.823   1.500  -0.875
ATOM    113  H3  TYR     7       2.823   1.500   0.875
ATOM    114  CA  TYR     7       3.970   2.846  -0.000
ATOM    115  HA  TYR     7       3.672   3.400  -0.890
ATOM    116  CB  TYR     7       3.577   3.654   1.232
ATOM    117  HB2 TYR     7       2.497   3.801   1.241
ATOM    118  HB3 TYR     7       3.877   3.116   2.131
ATOM    119  CG  TYR     7       4.267   4.996   1.195
ATOM    120  CD1 TYR     7       4.060   5.919   2.227
ATOM    121  HD1 TYR     7       3.400   5.668   3.058
ATOM    122  CE1 TYR     7       4.700   7.164   2.193
ATOM    123  HE1 TYR     7       4.539   7.882   2.997
ATOM    124  CZ  TYR     7       5.547   7.486   1.126
ATOM    125  OH  TYR     7       6.169   8.695   1.092
ATOM    126  HH  TYR     7       5.956   9.247   1.848
ATOM    127  CE2 TYR     7       5.755   6.563   0.094
ATOM    128  HE2 TYR     7       6.415   6.814  -0.737
ATOM    129  CD2 TYR     7       5.115   5.318   0.128
ATOM    130  HD2 TYR     7       5.276   4.600  -0.676
ATOM    131  C   TYR     7       5.486   2.705  -0.000
ATOM    132  O   TYR     7       6.009   1.593  -0.000
ATOM    133  N   VAL     8       6.191   3.839  -0.000
ATOM    134  H   VAL     8       5.715   4.730  -0.000
ATOM    135  CA  VAL     8       7.640   3.839  -0.000
ATOM    136  HA  VAL     8       8.004   3.325   0.890
ATOM    137  CB  VAL     8       8.189   3.127  -1.232
ATOM    138  HB  VAL     8       7.841   2.094  -1.241
ATOM    139  CG1 VAL     8       7.701   3.839  -2.490
ATOM    140 HG11 VAL     8       8.050   4.872  -2.481
ATOM    141 HG12 VAL     8       8.093   3.331  -3.371
ATOM    142 HG13 VAL     8       6.612   3.824  -2.517
ATOM    143  CG2 VAL     8       9.713   3.149  -1.195
ATOM    144 HG21 VAL     8      10.062   2.640  -0.296
ATOM    145 HG22 VAL     8      10.106   2.641  -2.075
ATOM    146 HG23 VAL     8      10.062   4.181  -1.186
ATOM    147  C   VAL     8       8.188   5.259  -0.000
ATOM    148  O   VAL     8       7.425   6.222   0.000
ATOM    149  N   GLY     9       9.517   5.386  -0.000
ATOM    150  H   GLY     9      10.103   4.564  -0.000
ATOM    151  CA  GLY     9      10.161   6.684   0.000
ATOM    152  HA2 GLY     9       9.863   7.239  -0.890
ATOM    153  HA3 GLY     9       9.863   7.239   0.890
ATOM    154  C   GLY     9      11.675   6.525  -0.000
ATOM    155  O   GLY     9      12.184   5.407  -0.000
ATOM    156  OXT GLY     9      12.406   7.513  -0.000
TER
"""


class TestAtomIndexingOneBased(unittest.TestCase):
    def setUp(self):
        p1 = AmberSubSystemFromSequence("NALA ALA CALA")
        p2 = AmberSubSystemFromSequence("NALA ALA CALA")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        self.system = b.build_system([p1, p2]).finalize()

    def test_absolute_index(self):
        self.assertEqual(self.system.index.atom(1, "CA", one_based=True), 4)
        self.assertEqual(self.system.index.atom(2, "CA", one_based=True), 14)
        self.assertEqual(self.system.index.atom(3, "CA", one_based=True), 24)
        self.assertEqual(self.system.index.atom(4, "CA", one_based=True), 37)
        self.assertEqual(self.system.index.atom(5, "CA", one_based=True), 47)
        self.assertEqual(self.system.index.atom(6, "CA", one_based=True), 57)

    def test_relative_index(self):
        self.assertEqual(self.system.index.atom(1, "CA", chainid=1, one_based=True), 4)
        self.assertEqual(self.system.index.atom(2, "CA", chainid=1, one_based=True), 14)
        self.assertEqual(self.system.index.atom(3, "CA", chainid=1, one_based=True), 24)
        self.assertEqual(self.system.index.atom(1, "CA", chainid=2, one_based=True), 37)
        self.assertEqual(self.system.index.atom(2, "CA", chainid=2, one_based=True), 47)
        self.assertEqual(self.system.index.atom(3, "CA", chainid=2, one_based=True), 57)


class TestAtomIndexingZeroBased(unittest.TestCase):
    def setUp(self):
        p1 = AmberSubSystemFromSequence("NALA ALA CALA")
        p2 = AmberSubSystemFromSequence("NALA ALA CALA")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        self.system = b.build_system([p1, p2]).finalize()

    def test_absolute_index(self):
        self.assertEqual(self.system.index.atom(0, "CA", one_based=False), 4)
        self.assertEqual(self.system.index.atom(1, "CA", one_based=False), 14)
        self.assertEqual(self.system.index.atom(2, "CA", one_based=False), 24)
        self.assertEqual(self.system.index.atom(3, "CA", one_based=False), 37)
        self.assertEqual(self.system.index.atom(4, "CA", one_based=False), 47)
        self.assertEqual(self.system.index.atom(5, "CA", one_based=False), 57)

    def test_relative_index(self):
        self.assertEqual(self.system.index.atom(0, "CA", chainid=0, one_based=False), 4)
        self.assertEqual(
            self.system.index.atom(1, "CA", chainid=0, one_based=False), 14
        )
        self.assertEqual(
            self.system.index.atom(2, "CA", chainid=0, one_based=False), 24
        )
        self.assertEqual(
            self.system.index.atom(0, "CA", chainid=1, one_based=False), 37
        )
        self.assertEqual(
            self.system.index.atom(1, "CA", chainid=1, one_based=False), 47
        )
        self.assertEqual(
            self.system.index.atom(2, "CA", chainid=1, one_based=False), 57
        )

    def test_expected_resname_mismatch_should_raise(self):
        with self.assertRaises(KeyError):
            # This residue is an alanine
            self.system.index.atom(1, "CA", expected_resname="CYS")

    def test_expected_resname_should_handle_n_term(self):
        self.assertEqual(
            self.system.index.atom(
                0, "CA", expected_resname="ALA", chainid=0, one_based=False
            ),
            4,
        )

    def test_expected_resname_should_handle_c_term(self):
        self.assertEqual(
            self.system.index.atom(
                2, "CA", expected_resname="ALA", chainid=0, one_based=False
            ),
            24,
        )


class TestAtomIndexingExpectedResname(unittest.TestCase):
    def setUp(self):
        p1 = AmberSubSystemFromSequence("NALA CYS CLYS")
        p2 = AmberSubSystemFromSequence("NARG TRP CTYR")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        self.system = b.build_system([p1, p2]).finalize()

    def test_expected_resname_mismatch_should_raise(self):
        with self.assertRaises(KeyError):
            # This residue is CYS, not a VAL
            self.system.index.atom(1, "CA", expected_resname="VAL", one_based=False)

    def test_expected_resname_should_match(self):
        self.system.index.atom(0, "CA", expected_resname="ALA", one_based=False)
        self.system.index.atom(1, "CA", expected_resname="CYS", one_based=False)
        self.system.index.atom(2, "CA", expected_resname="LYS", one_based=False)
        self.system.index.atom(3, "CA", expected_resname="ARG", one_based=False)
        self.system.index.atom(4, "CA", expected_resname="TRP", one_based=False)
        self.system.index.atom(5, "CA", expected_resname="TYR", one_based=False)


class TestAtomIndexingHistidine(unittest.TestCase):
    def test_hid_should_match_his(self):
        p1 = AmberSubSystemFromSequence("NALA HID CLYS")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        system = b.build_system([p1]).finalize()
        system.index.atom(1, "CA", expected_resname="HIS", one_based=False)

    def test_hie_should_match_his(self):
        p1 = AmberSubSystemFromSequence("NALA HIE CLYS")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        system = b.build_system([p1]).finalize()
        system.index.atom(1, "CA", expected_resname="HIS", one_based=False)

    def test_hip_should_match_his(self):
        p1 = AmberSubSystemFromSequence("NALA HIP CLYS")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        system = b.build_system([p1]).finalize()
        system.index.atom(1, "CA", expected_resname="HIS", one_based=False)


class TestAtomIndexingAsparticAcid(unittest.TestCase):
    def test_ash_should_match_asp(self):
        p1 = AmberSubSystemFromSequence("NALA ASH CLYS")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        system = b.build_system([p1]).finalize()
        system.index.atom(1, "CA", expected_resname="ASP", one_based=False)


class TestAtomIndexingGlutamicAcid(unittest.TestCase):
    def test_ash_should_match_asp(self):
        p1 = AmberSubSystemFromSequence("NALA GLH CLYS")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        system = b.build_system([p1]).finalize()
        system.index.atom(1, "CA", expected_resname="GLU", one_based=False)


class TestAtomIndexingLysine(unittest.TestCase):
    def test_lyn_should_match_lys(self):
        p1 = AmberSubSystemFromSequence("NALA LYN CLYS")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        system = b.build_system([p1]).finalize()
        system.index.atom(1, "CA", expected_resname="LYS", one_based=False)


class TestAtomIndexingDisulfide(unittest.TestCase):
    def test_lyn_should_match_lys(self):
        p = AmberSubSystemFromSequence("NCYX ALA CCYX")
        p.add_disulfide(ResidueIndex(0), ResidueIndex(2))
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        system = b.build_system([p]).finalize()
        system.index.atom(0, "CA", expected_resname="CYS", one_based=False)
        system.index.atom(2, "CA", expected_resname="CYS", one_based=False)


class TestResidueIndexingOneBased(unittest.TestCase):
    def setUp(self):
        p1 = AmberSubSystemFromSequence("NALA ALA CALA")
        p2 = AmberSubSystemFromSequence("NALA ALA CALA")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        self.system = b.build_system([p1, p2]).finalize()

    def test_absolute_index(self):
        self.assertEqual(self.system.index.residue(1, one_based=True), 0)
        self.assertEqual(self.system.index.residue(2, one_based=True), 1)
        self.assertEqual(self.system.index.residue(3, one_based=True), 2)
        self.assertEqual(self.system.index.residue(4, one_based=True), 3)
        self.assertEqual(self.system.index.residue(5, one_based=True), 4)
        self.assertEqual(self.system.index.residue(6, one_based=True), 5)

    def test_relative_index(self):
        self.assertEqual(self.system.index.residue(1, chainid=1, one_based=True), 0)
        self.assertEqual(self.system.index.residue(2, chainid=1, one_based=True), 1)
        self.assertEqual(self.system.index.residue(3, chainid=1, one_based=True), 2)
        self.assertEqual(self.system.index.residue(1, chainid=2, one_based=True), 3)
        self.assertEqual(self.system.index.residue(2, chainid=2, one_based=True), 4)
        self.assertEqual(self.system.index.residue(3, chainid=2, one_based=True), 5)


class TestResidueIndexingZeroBased(unittest.TestCase):
    def setUp(self):
        p1 = AmberSubSystemFromSequence("NALA ALA CALA")
        p2 = AmberSubSystemFromSequence("NALA ALA CALA")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        self.system = b.build_system([p1, p2]).finalize()

    def test_absolute_index(self):
        self.assertEqual(self.system.index.residue(0, one_based=False), 0)
        self.assertEqual(self.system.index.residue(1, one_based=False), 1)
        self.assertEqual(self.system.index.residue(2, one_based=False), 2)
        self.assertEqual(self.system.index.residue(3, one_based=False), 3)
        self.assertEqual(self.system.index.residue(4, one_based=False), 4)
        self.assertEqual(self.system.index.residue(5, one_based=False), 5)

    def test_relative_index(self):
        self.assertEqual(self.system.index.residue(0, chainid=0, one_based=False), 0)
        self.assertEqual(self.system.index.residue(1, chainid=0, one_based=False), 1)
        self.assertEqual(self.system.index.residue(2, chainid=0, one_based=False), 2)
        self.assertEqual(self.system.index.residue(0, chainid=1, one_based=False), 3)
        self.assertEqual(self.system.index.residue(1, chainid=1, one_based=False), 4)
        self.assertEqual(self.system.index.residue(2, chainid=1, one_based=False), 5)


class TestResidueIndexingExpectedResname(unittest.TestCase):
    def setUp(self):
        p1 = AmberSubSystemFromSequence("NALA CYS CLYS")
        p2 = AmberSubSystemFromSequence("NARG TRP CTYR")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        self.system = b.build_system([p1, p2]).finalize()

    def test_expected_resname_mismatch_should_raise(self):
        with self.assertRaises(KeyError):
            # This residue is CYS, not a VAL
            self.system.index.residue(1, expected_resname="VAL", one_based=False)

    def test_expected_resname_should_match(self):
        self.system.index.residue(0, expected_resname="ALA", one_based=False)
        self.system.index.residue(1, expected_resname="CYS", one_based=False)
        self.system.index.residue(2, expected_resname="LYS", one_based=False)
        self.system.index.residue(3, expected_resname="ARG", one_based=False)
        self.system.index.residue(4, expected_resname="TRP", one_based=False)
        self.system.index.residue(5, expected_resname="TYR", one_based=False)


class TestResidueIndexingHistidine(unittest.TestCase):
    def test_hid_should_match_his(self):
        p1 = AmberSubSystemFromSequence("NALA HID CLYS")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        system = b.build_system([p1]).finalize()
        system.index.residue(1, expected_resname="HIS", one_based=False)

    def test_hie_should_match_his(self):
        p1 = AmberSubSystemFromSequence("NALA HIE CLYS")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        system = b.build_system([p1]).finalize()
        system.index.residue(1, expected_resname="HIS", one_based=False)

    def test_hip_should_match_his(self):
        p1 = AmberSubSystemFromSequence("NALA HIP CLYS")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        system = b.build_system([p1]).finalize()
        system.index.residue(1, expected_resname="HIS", one_based=False)


class TestResidueIndexingAsparticAcid(unittest.TestCase):
    def test_ash_should_match_asp(self):
        p1 = AmberSubSystemFromSequence("NALA ASH CLYS")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        system = b.build_system([p1]).finalize()
        system.index.residue(1, expected_resname="ASP", one_based=False)


class TestResidueIndexingGlutamicAcid(unittest.TestCase):
    def test_ash_should_match_asp(self):
        p1 = AmberSubSystemFromSequence("NALA GLH CLYS")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        system = b.build_system([p1]).finalize()
        system.index.residue(1, expected_resname="GLU", one_based=False)


class TestResidueIndexingLysine(unittest.TestCase):
    def test_lyn_should_match_lys(self):
        p1 = AmberSubSystemFromSequence("NALA LYN CLYS")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        system = b.build_system([p1]).finalize()
        system.index.residue(1, expected_resname="LYS", one_based=False)


class TestResidueIndexingDisulfide(unittest.TestCase):
    def test_lyn_should_match_lys(self):
        p = AmberSubSystemFromSequence("NCYX ALA CCYX")
        p.add_disulfide(ResidueIndex(0), ResidueIndex(2))
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        system = b.build_system([p]).finalize()
        system.index.residue(0, expected_resname="CYS", one_based=False)
        system.index.residue(2, expected_resname="CYS", one_based=False)


class TestAtomIndexingFromPDB(unittest.TestCase):
    def setUp(self):
        with in_temp_dir():
            with open("pdb.pdb", "w") as outfile:
                outfile.write(pdb_string)
            p = AmberSubSystemFromPdbFile("pdb.pdb")
            options = AmberOptions()
            b = AmberSystemBuilder(options)
            self.system = b.build_system([p]).finalize()

    def test_absolute_zero_based(self):
        index1 = self.system.index.atom(0, "CA", expected_resname="ALA")
        index2 = self.system.index.atom(1, "CA", expected_resname="CYS")
        index3 = self.system.index.atom(2, "CA", expected_resname="ASP")
        index4 = self.system.index.atom(3, "CA", expected_resname="ARG")
        index5 = self.system.index.atom(4, "CA", expected_resname="LYS")
        index6 = self.system.index.atom(5, "CA", expected_resname="TRP")
        index7 = self.system.index.atom(6, "CA", expected_resname="TYR")
        index8 = self.system.index.atom(7, "CA", expected_resname="VAL")
        index9 = self.system.index.atom(8, "CA", expected_resname="GLY")

        self.assertEqual(index1, 4)
        self.assertEqual(index2, 14)
        self.assertEqual(index3, 25)
        self.assertEqual(index4, 40)
        self.assertEqual(index5, 64)
        self.assertEqual(index6, 86)
        self.assertEqual(index7, 113)
        self.assertEqual(index8, 134)
        self.assertEqual(index9, 150)

    def test_absolute_one_based(self):
        index1 = self.system.index.atom(1, "CA", expected_resname="ALA", one_based=True)
        index2 = self.system.index.atom(2, "CA", expected_resname="CYS", one_based=True)
        index3 = self.system.index.atom(3, "CA", expected_resname="ASP", one_based=True)
        index4 = self.system.index.atom(4, "CA", expected_resname="ARG", one_based=True)
        index5 = self.system.index.atom(5, "CA", expected_resname="LYS", one_based=True)
        index6 = self.system.index.atom(6, "CA", expected_resname="TRP", one_based=True)
        index7 = self.system.index.atom(7, "CA", expected_resname="TYR", one_based=True)
        index8 = self.system.index.atom(8, "CA", expected_resname="VAL", one_based=True)
        index9 = self.system.index.atom(9, "CA", expected_resname="GLY", one_based=True)

        self.assertEqual(index1, 4)
        self.assertEqual(index2, 14)
        self.assertEqual(index3, 25)
        self.assertEqual(index4, 40)
        self.assertEqual(index5, 64)
        self.assertEqual(index6, 86)
        self.assertEqual(index7, 113)
        self.assertEqual(index8, 134)
        self.assertEqual(index9, 150)

    def test_relative_zero_based(self):
        index1 = self.system.index.atom(0, "CA", chainid=0, expected_resname="TYR")
        index2 = self.system.index.atom(1, "CA", chainid=0, expected_resname="VAL")
        index3 = self.system.index.atom(2, "CA", chainid=0, expected_resname="GLY")
        index4 = self.system.index.atom(0, "CA", chainid=1, expected_resname="ALA")
        index5 = self.system.index.atom(1, "CA", chainid=1, expected_resname="CYS")
        index6 = self.system.index.atom(2, "CA", chainid=1, expected_resname="ASP")
        index7 = self.system.index.atom(0, "CA", chainid=2, expected_resname="ARG")
        index8 = self.system.index.atom(1, "CA", chainid=2, expected_resname="LYS")
        index9 = self.system.index.atom(2, "CA", chainid=2, expected_resname="TRP")

        self.assertEqual(index1, 113)
        self.assertEqual(index2, 134)
        self.assertEqual(index3, 150)
        self.assertEqual(index4, 4)
        self.assertEqual(index5, 14)
        self.assertEqual(index6, 25)
        self.assertEqual(index7, 40)
        self.assertEqual(index8, 64)
        self.assertEqual(index9, 86)

    def test_relative_one_based(self):
        index1 = self.system.index.atom(
            1, "CA", chainid=1, expected_resname="TYR", one_based=True
        )
        index2 = self.system.index.atom(
            2, "CA", chainid=1, expected_resname="VAL", one_based=True
        )
        index3 = self.system.index.atom(
            3, "CA", chainid=1, expected_resname="GLY", one_based=True
        )
        index4 = self.system.index.atom(
            1, "CA", chainid=2, expected_resname="ALA", one_based=True
        )
        index5 = self.system.index.atom(
            2, "CA", chainid=2, expected_resname="CYS", one_based=True
        )
        index6 = self.system.index.atom(
            3, "CA", chainid=2, expected_resname="ASP", one_based=True
        )
        index7 = self.system.index.atom(
            1, "CA", chainid=3, expected_resname="ARG", one_based=True
        )
        index8 = self.system.index.atom(
            2, "CA", chainid=3, expected_resname="LYS", one_based=True
        )
        index9 = self.system.index.atom(
            3, "CA", chainid=3, expected_resname="TRP", one_based=True
        )

        self.assertEqual(index1, 113)
        self.assertEqual(index2, 134)
        self.assertEqual(index3, 150)
        self.assertEqual(index4, 4)
        self.assertEqual(index5, 14)
        self.assertEqual(index6, 25)
        self.assertEqual(index7, 40)
        self.assertEqual(index8, 64)
        self.assertEqual(index9, 86)

    def test_mismatch_should_raise(self):
        with self.assertRaises(KeyError):
            # Residue is TYR, not GLU
            self.system.index.atom(0, "CA", chainid=0, expected_resname="GLU")


class TestAtomIndexingFromPDBExplicit(unittest.TestCase):
    def setUp(self):
        with in_temp_dir():
            with open("pdb.pdb", "w") as outfile:
                outfile.write(pdb_string)
            p = AmberSubSystemFromPdbFile("pdb.pdb")
            options = AmberOptions(
                solvation="explicit",
                enable_pme=True,
                enable_pressure_coupling=True,
                cutoff=0.9,
            )
            b = AmberSystemBuilder(options)
            self.system = b.build_system([p]).finalize()

    def test_absolute(self):
        index1 = self.system.index.atom(0, "CA", expected_resname="ALA")
        index2 = self.system.index.atom(1, "CA", expected_resname="CYS")
        index3 = self.system.index.atom(2, "CA", expected_resname="ASP")
        index4 = self.system.index.atom(3, "CA", expected_resname="ARG")
        index5 = self.system.index.atom(4, "CA", expected_resname="LYS")
        index6 = self.system.index.atom(5, "CA", expected_resname="TRP")
        index7 = self.system.index.atom(6, "CA", expected_resname="TYR")
        index8 = self.system.index.atom(7, "CA", expected_resname="VAL")
        index9 = self.system.index.atom(8, "CA", expected_resname="GLY")

        self.assertEqual(index1, 4)
        self.assertEqual(index2, 14)
        self.assertEqual(index3, 25)
        self.assertEqual(index4, 40)
        self.assertEqual(index5, 64)
        self.assertEqual(index6, 86)
        self.assertEqual(index7, 113)
        self.assertEqual(index8, 134)
        self.assertEqual(index9, 150)

    def test_relative(self):
        index1 = self.system.index.atom(0, "CA", chainid=0, expected_resname="TYR")
        index2 = self.system.index.atom(1, "CA", chainid=0, expected_resname="VAL")
        index3 = self.system.index.atom(2, "CA", chainid=0, expected_resname="GLY")
        index4 = self.system.index.atom(0, "CA", chainid=1, expected_resname="ALA")
        index5 = self.system.index.atom(1, "CA", chainid=1, expected_resname="CYS")
        index6 = self.system.index.atom(2, "CA", chainid=1, expected_resname="ASP")
        index7 = self.system.index.atom(0, "CA", chainid=2, expected_resname="ARG")
        index8 = self.system.index.atom(1, "CA", chainid=2, expected_resname="LYS")
        index9 = self.system.index.atom(2, "CA", chainid=2, expected_resname="TRP")

        self.assertEqual(index1, 113)
        self.assertEqual(index2, 134)
        self.assertEqual(index3, 150)
        self.assertEqual(index4, 4)
        self.assertEqual(index5, 14)
        self.assertEqual(index6, 25)
        self.assertEqual(index7, 40)
        self.assertEqual(index8, 64)
        self.assertEqual(index9, 86)

    def test_extra_residue_indices_match(self):
        index1 = self.system.index.residue(9, expected_resname="HOH")
        index2 = self.system.index.residue(0, chainid=3, expected_resname="HOH")
        index3 = self.system.index.residue(422, expected_resname="HOH")
        index4 = self.system.index.residue(413, chainid=3, expected_resname="HOH")
        self.assertEqual(index1, index2)
        self.assertEqual(index3, index4)
