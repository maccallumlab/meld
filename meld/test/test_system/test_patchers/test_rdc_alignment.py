import unittest
import openmm as mm  # type: ignore
from meld import (
    AmberSubSystemFromSequence,
    AmberSystemBuilder,
    AmberOptions,
    add_rdc_alignment,
)


class TestRDCPatcher(unittest.TestCase):
    def setUp(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        options = AmberOptions(implicit_solvent_model="gbNeck2")
        b = AmberSystemBuilder(options)
        self.spec = b.build_system([p])

    def test_integrator_should_have_correct_type(self):
        new_spec = add_rdc_alignment(self.spec, num_alignments=1)

        self.assertIsInstance(new_spec.integrator, mm.CustomIntegrator)

    def test_integrator_should_have_correct_num_alignments(self):
        EXPECTED = 1

        new_spec = add_rdc_alignment(self.spec, num_alignments=1)
        system = new_spec.finalize()

        self.assertEqual(system.num_alignments, EXPECTED)
