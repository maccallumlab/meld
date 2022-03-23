#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import unittest
from meld import parse
from meld import (
    AmberOptions,
    AmberSubSystemFromSequence,
    AmberSystemBuilder,
    add_rdc_alignment,
)
from meld.system import scalers
from meld.system import indexing


class TestGetAA1(unittest.TestCase):
    def test_can_parse_simple_sequence(self):
        content = "AAAA"

        result = parse.get_sequence_from_AA1(content=content)

        self.assertEqual(result, "NALA ALA ALA CALA")

    def test_can_handle_multiple_lines(self):
        content = "AA\nAA\n"

        result = parse.get_sequence_from_AA1(content=content)

        self.assertEqual(result, "NALA ALA ALA CALA")

    def test_can_handle_comments(self):
        content = "AA\n#comment\nAA\n"

        result = parse.get_sequence_from_AA1(content=content)

        self.assertEqual(result, "NALA ALA ALA CALA")

    def test_can_handle_all_residues(self):
        content = "ACDEFGHIKLMNPQRSTVWY"

        result = parse.get_sequence_from_AA1(content=content)

        self.assertEqual(
            result,
            "NALA CYS ASP GLU PHE GLY HIE ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP CTYR",
        )

    def test_should_raise_with_bad_residue(self):
        content = "AX"

        with self.assertRaises(RuntimeError):
            parse.get_sequence_from_AA1(content=content)


class TestGetAA3(unittest.TestCase):
    def test_can_parse_simple_sequence(self):
        content = "ALA ALA ALA ALA"

        result = parse.get_sequence_from_AA3(content=content)

        self.assertEqual(result, "NALA ALA ALA CALA")

    def test_can_handle_comment(self):
        content = "ALA ALA\n#comment\nALA ALA\n"

        result = parse.get_sequence_from_AA3(content=content)

        self.assertEqual(result, "NALA ALA ALA CALA")

    def test_can_handle_all_residues(self):
        content = "ALA CYS ASP GLU PHE GLY HIE ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR"

        result = parse.get_sequence_from_AA3(content=content)

        self.assertEqual(
            result,
            "NALA CYS ASP GLU PHE GLY HIE ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP CTYR",
        )

    def test_can_handle_alternate_protonation_states(self):
        content = "ALA ASH GLH LYN HIE HIP HID ALA"

        result = parse.get_sequence_from_AA3(content=content)

        self.assertEqual(result, "NALA ASH GLH LYN HIE HIP HID CALA")

    def test_raises_error_on_bad_residue(self):
        content = "ALA XXX ALA"

        with self.assertRaises(RuntimeError):
            parse.get_sequence_from_AA3(content=content)


class TestGetSecondaryString(unittest.TestCase):
    def test_can_parse_simple_input(self):
        content = "..HHEE.."

        results = parse._get_secondary_sequence(content=content)

        self.assertEqual(results, "..HHEE..")

    def test_can_parse_multiline_input(self):
        content = "..HH\nEE..\n"

        results = parse._get_secondary_sequence(content=content)

        self.assertEqual(results, "..HHEE..")

    def test_can_parse_input_with_comment(self):
        content = "..HH\n#comment\nEE..\n"

        results = parse._get_secondary_sequence(content=content)

    def test_raises_on_bad_input(self):
        content = ".HEx"

        with self.assertRaises(RuntimeError):
            parse._get_secondary_sequence(content=content)


class TestExtractSecondaryRuns(unittest.TestCase):
    def test_can_handle_single_run(self):
        content = "HHHHH"

        results = parse._extract_secondary_runs(
            content, ss_type="H", run_length=5, at_least=5, first_residue=0
        )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.start, 0)
        self.assertEqual(result.end, 5)

    def test_can_handle_single_run_plus_other(self):
        content = "...HHHHHEEE"

        results = parse._extract_secondary_runs(
            content, ss_type="H", run_length=5, at_least=5, first_residue=0
        )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.start, 3)
        self.assertEqual(result.end, 8)

    def test_can_handle_multiple_runs(self):
        content = "...HHHHH...HHHHH..."

        results = parse._extract_secondary_runs(
            content, ss_type="H", run_length=5, at_least=5, first_residue=0
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].start, 3)
        self.assertEqual(results[0].end, 8)
        self.assertEqual(results[1].start, 11)
        self.assertEqual(results[1].end, 16)

    def test_can_handle_at_least(self):
        content = "...HHHHH..."

        results = parse._extract_secondary_runs(
            content, ss_type="H", run_length=5, at_least=4, first_residue=0
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].start, 2)
        self.assertEqual(results[0].end, 7)
        self.assertEqual(results[1].start, 3)
        self.assertEqual(results[1].end, 8)
        self.assertEqual(results[2].start, 4)
        self.assertEqual(results[2].end, 9)


class TestGetSecondaryStructures(unittest.TestCase):
    def setUp(self):
        p = AmberSubSystemFromSequence("NALA ALA ALA ALA ALA ALA ALA ALA ALA CALA")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        self.system = b.build_system([p])
        self.ss = "HHHHHEEEEE"
        self.scaler = scalers.LinearScaler(0, 1)

    def test_adds_correct_number_of_groups(self):
        results = parse.get_secondary_structure_restraints(
            content=self.ss, system=self.system, scaler=self.scaler
        )

        self.assertEqual(len(results), 4)

    def each_group_should_have_six_restraints(self):
        results = parse.get_secondary_structure_restraints(
            content=self.ss, system=self.system, scaler=self.scaler
        )

        for group in results:
            self.assertEqual(len(group), 6)


class TestGetRdcRestraints(unittest.TestCase):
    def setUp(self):
        p = AmberSubSystemFromSequence("NALA ALA ALA ALA ALA ALA ALA ALA ALA CALA")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        self.scaler = scalers.LinearScaler(0, 1)
        spec = b.build_system([p])
        spec = add_rdc_alignment(spec, num_alignments=1)
        self.system = spec.finalize()

    def test_ignores_comment_lines(self):
        contents = "#\n#\n"
        results = parse.get_rdc_restraints(
            content=contents,
            system=self.system,
            scaler=self.scaler,
            alignment_index=0,
        )

        self.assertEqual(len(results), 0)

    def test_adds_correct_number_of_restraints(self):
        contents = "#\n2 N 2 H 10 0 10 25000 1.0 1\n"
        results = parse.get_rdc_restraints(
            content=contents,
            system=self.system,
            scaler=self.scaler,
            alignment_index=0,
        )

        self.assertEqual(len(results), 1)
