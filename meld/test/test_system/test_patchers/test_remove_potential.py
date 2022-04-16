import unittest
from meld import (
    AmberSubSystemFromSequence,
    AmberSystemBuilder,
    AmberOptions,
    remove_potential,
    unit,
)


class TestRemovePotential(unittest.TestCase):
    def setUp(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        options = AmberOptions(implicit_solvent_model="gbNeck2")
        b = AmberSystemBuilder(options)
        self.spec = b.build_system([p])

    def test_should_freeze_all_atoms(self):
        new_spec = remove_potential(self.spec)

        for i in range(self.spec.system.getNumParticles()):
            mass = new_spec.system.getParticleMass(i).value_in_unit(unit.dalton)
            self.assertEqual(mass, 0.0)

    def test_should_remove_all_forces(self):
        new_spec = remove_potential(self.spec)

        n_forces = new_spec.system.getNumForces()

        self.assertEqual(n_forces, 0)
