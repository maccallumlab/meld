import unittest
from meld import (
    AmberSubSystemFromSequence,
    AmberSystemBuilder,
    AmberOptions,
    add_rdc_alignment,
)
from meld.system.patchers.rdc_integrator import CustomRDCIntegrator


class TestRDCPatcher(unittest.TestCase):
    def setUp(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        options = AmberOptions(implicit_solvent_model="gbNeck2")
        b = AmberSystemBuilder(options)
        self.spec = b.build_system([p])

    def test_integrator_should_have_correct_type(self):
        new_spec = add_rdc_alignment(self.spec, num_alignments=1)

        self.assertIsInstance(new_spec.integrator, CustomRDCIntegrator)

    def test_integrator_should_have_correct_num_alignments(self):
        EXPECTED = 1

        new_spec = add_rdc_alignment(self.spec, num_alignments=1)

        self.assertEqual(new_spec.integrator.num_alignments, EXPECTED)
