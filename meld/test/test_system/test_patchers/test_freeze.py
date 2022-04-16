import unittest
from meld import (
    AmberSubSystemFromSequence,
    AmberSystemBuilder,
    AmberOptions,
    freeze_atoms,
    unit,
)


class TestFreeze(unittest.TestCase):
    def setUp(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        options = AmberOptions(implicit_solvent_model="gbNeck2")
        b = AmberSystemBuilder(options)
        self.spec = b.build_system([p])

    def test_should_freeze_all_by_default(self):
        new_spec = freeze_atoms(self.spec)

        for i in range(self.spec.system.getNumParticles()):
            mass = new_spec.system.getParticleMass(i).value_in_unit(unit.dalton)
            self.assertEqual(mass, 0.0)

    def test_should_freeze_specific_atom(self):
        atom_index = self.spec.index.atom(1, "CA")
        new_spec = freeze_atoms(self.spec, atoms=[atom_index])

        for i in range(self.spec.system.getNumParticles()):
            mass = new_spec.system.getParticleMass(i).value_in_unit(unit.dalton)
            if i == atom_index:
                self.assertEqual(mass, 0.0)
            else:
                self.assertNotEqual(mass, 0.0)
