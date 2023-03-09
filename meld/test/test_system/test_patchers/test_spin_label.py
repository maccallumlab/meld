import unittest
import openmm as mm  # type: ignore
from meld import (
    AmberSubSystemFromSequence,
    AmberSystemBuilder,
    AmberOptions,
    add_virtual_spin_label,
)


class TestSpinLabelPatcher(unittest.TestCase):
    def setUp(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        options = AmberOptions(implicit_solvent_model="gbNeck2")
        b = AmberSystemBuilder(options)
        self.spec = b.build_system([p])

    def test_should_add_particle(self):
        orig_particles = self.spec.system.getNumParticles()
        EXPECTED = orig_particles + 1

        new_spec = add_virtual_spin_label(self.spec, self.spec.index.residue(1), "OND")
        system_particles = new_spec.system.getNumParticles()

        self.assertEqual(system_particles, EXPECTED)

    def test_should_add_nb_particle(self):
        orig_particles = self.spec.system.getNumParticles()
        EXPECTED = orig_particles + 1

        new_spec = add_virtual_spin_label(self.spec, self.spec.index.residue(1), "OND")
        system_particles = _get_nb_force(new_spec.system).getNumParticles()

        self.assertEqual(system_particles, EXPECTED)

    def test_should_add_gb_particle(self):
        orig_particles = self.spec.system.getNumParticles()
        EXPECTED = orig_particles + 1

        new_spec = add_virtual_spin_label(self.spec, self.spec.index.residue(1), "OND")
        system_particles = _get_customgb_force(new_spec.system).getNumParticles()

        self.assertEqual(system_particles, EXPECTED)

    def test_should_have_correct_indexing(self):
        new_spec = add_virtual_spin_label(self.spec, self.spec.index.residue(1), "OND")
        system = new_spec.finalize()
        # It should be one after the O of the second residue
        EXPECTED = system.index.atom(1, "O") + 1

        actaul = system.index.atom(1, "OND")

        self.assertEqual(actaul, EXPECTED)


class TestSpinLabelPatcherOBC(unittest.TestCase):
    def setUp(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        options = AmberOptions(implicit_solvent_model="obc")
        b = AmberSystemBuilder(options)
        self.spec = b.build_system([p])

    def test_should_add_particle(self):
        orig_particles = self.spec.system.getNumParticles()
        EXPECTED = orig_particles + 1

        new_spec = add_virtual_spin_label(self.spec, self.spec.index.residue(1), "OND")
        system_particles = new_spec.system.getNumParticles()

        self.assertEqual(system_particles, EXPECTED)

    def test_should_add_nb_particle(self):
        orig_particles = self.spec.system.getNumParticles()
        EXPECTED = orig_particles + 1

        new_spec = add_virtual_spin_label(self.spec, self.spec.index.residue(1), "OND")
        system_particles = _get_nb_force(new_spec.system).getNumParticles()

        self.assertEqual(system_particles, EXPECTED)

    def test_should_add_gb_particle(self):
        orig_particles = self.spec.system.getNumParticles()
        EXPECTED = orig_particles + 1

        new_spec = add_virtual_spin_label(self.spec, self.spec.index.residue(1), "OND")
        system_particles = _get_obc_force(new_spec.system).getNumParticles()

        self.assertEqual(system_particles, EXPECTED)

    def test_should_have_correct_indexing(self):
        new_spec = add_virtual_spin_label(self.spec, self.spec.index.residue(1), "OND")
        system = new_spec.finalize()
        # It should be one after the O of the second residue
        EXPECTED = system.index.atom(1, "O") + 1

        actaul = system.index.atom(1, "OND")

        self.assertEqual(actaul, EXPECTED)


class TestSpinLabelPatcherTwice(unittest.TestCase):
    def setUp(self):
        p = AmberSubSystemFromSequence("NALA ALA ALA CALA")
        options = AmberOptions(implicit_solvent_model="gbNeck2")
        b = AmberSystemBuilder(options)
        self.spec = b.build_system([p])

    def test_should_add_particle(self):
        orig_particles = self.spec.system.getNumParticles()
        EXPECTED = orig_particles + 2

        new_spec = add_virtual_spin_label(self.spec, self.spec.index.residue(1), "OND")
        new_spec = add_virtual_spin_label(new_spec, self.spec.index.residue(2), "OND")
        system_particles = new_spec.system.getNumParticles()

        self.assertEqual(system_particles, EXPECTED)

    def test_should_add_nb_particle(self):
        orig_particles = self.spec.system.getNumParticles()
        EXPECTED = orig_particles + 2

        new_spec = add_virtual_spin_label(self.spec, self.spec.index.residue(1), "OND")
        new_spec = add_virtual_spin_label(new_spec, self.spec.index.residue(2), "OND")
        system_particles = _get_nb_force(new_spec.system).getNumParticles()

        self.assertEqual(system_particles, EXPECTED)

    def test_should_add_gb_particle(self):
        orig_particles = self.spec.system.getNumParticles()
        EXPECTED = orig_particles + 2

        new_spec = add_virtual_spin_label(self.spec, self.spec.index.residue(1), "OND")
        new_spec = add_virtual_spin_label(new_spec, self.spec.index.residue(2), "OND")
        system_particles = _get_customgb_force(new_spec.system).getNumParticles()

        self.assertEqual(system_particles, EXPECTED)

    def test_should_have_correct_indexing(self):
        new_spec = add_virtual_spin_label(self.spec, self.spec.index.residue(1), "OND")
        new_spec = add_virtual_spin_label(new_spec, self.spec.index.residue(2), "OND")
        system = new_spec.finalize()
        # It should be one after the O of the second residue
        EXPECTED1 = system.index.atom(1, "O") + 1
        EXPECTED2 = system.index.atom(2, "O") + 1

        actual1 = system.index.atom(1, "OND")
        actual2 = system.index.atom(2, "OND")

        self.assertEqual(actual1, EXPECTED1)
        self.assertEqual(actual2, EXPECTED2)


def _get_nb_force(system):
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            return force
    raise RuntimeError("No NonbondedForce found")


def _get_customgb_force(system):
    for force in system.getForces():
        if isinstance(force, mm.CustomGBForce):
            return force
    raise RuntimeError("No CustomGBForce found")


def _get_obc_force(system):
    for force in system.getForces():
        if isinstance(force, mm.GBSAOBCForce):
            return force
    raise RuntimeError("No CustomGBForce found")
