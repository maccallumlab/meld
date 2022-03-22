import unittest
import openmm as mm  # type: ignore
from meld import AmberSubSystemFromSequence, AmberSystemBuilder, add_rdc_alignment
from meld.system import indexing


class TestGetRdcRestraintsNeck(unittest.TestCase):
    def setUp(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        b = AmberSystemBuilder()
        self.spec = b.build_system([p], implicit_solvent_model="gbNeck2")

    def test_should_add_particles(self):
        orig_particles = self.spec.system.getNumParticles()
        EXPECTED = orig_particles + 2

        new_spec, _ = add_rdc_alignment(self.spec)
        system_particles = new_spec.system.getNumParticles()
        nb_particles = _get_nb_force(new_spec.system).getNumParticles()
        gb_particles = _get_customgb_force(new_spec.system).getNumParticles()

        self.assertEqual(system_particles, EXPECTED)
        self.assertEqual(nb_particles, EXPECTED)
        self.assertEqual(gb_particles, EXPECTED)

    def test_should_add_residues(self):
        n_orig_residues = self.spec.topology.getNumResidues()
        EXPECTED = n_orig_residues + 1

        new_spec, _ = add_rdc_alignment(self.spec)
        n_residues = new_spec.topology.getNumResidues()

        self.assertEqual(n_residues, EXPECTED)

    def test_should_be_last_residue(self):
        n_orig_residues = self.spec.topology.getNumResidues()
        EXPECTED = indexing.ResidueIndex(n_orig_residues)

        _, res_index = add_rdc_alignment(self.spec)

        self.assertEqual(res_index, EXPECTED)


class TestGetRdcRestraintsOBC(unittest.TestCase):
    def setUp(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        b = AmberSystemBuilder()
        self.spec = b.build_system([p], implicit_solvent_model="obc")

    def test_should_add_particles(self):
        orig_particles = self.spec.system.getNumParticles()
        EXPECTED = orig_particles + 2

        new_spec, _ = add_rdc_alignment(self.spec)
        system_particles = new_spec.system.getNumParticles()
        nb_particles = _get_nb_force(new_spec.system).getNumParticles()
        gb_particles = _get_obc_force(new_spec.system).getNumParticles()

        self.assertEqual(system_particles, EXPECTED)
        self.assertEqual(nb_particles, EXPECTED)
        self.assertEqual(gb_particles, EXPECTED)

    def test_should_add_residues(self):
        n_orig_residues = self.spec.topology.getNumResidues()
        EXPECTED = n_orig_residues + 1

        new_spec, _ = add_rdc_alignment(self.spec)
        n_residues = new_spec.topology.getNumResidues()

        self.assertEqual(n_residues, EXPECTED)

    def test_should_be_last_residue(self):
        n_orig_residues = self.spec.topology.getNumResidues()
        EXPECTED = indexing.ResidueIndex(n_orig_residues)

        _, res_index = add_rdc_alignment(self.spec)

        self.assertEqual(res_index, EXPECTED)


class TestGetRdcRestraintsTwice(unittest.TestCase):
    def setUp(self):
        p = AmberSubSystemFromSequence("NALA ALA CALA")
        b = AmberSystemBuilder()
        self.spec = b.build_system([p], implicit_solvent_model="gbNeck2")

    def test_should_add_particles(self):
        orig_particles = self.spec.system.getNumParticles()
        EXPECTED = orig_particles + 4

        new_spec, _ = add_rdc_alignment(self.spec)
        new_spec, _ = add_rdc_alignment(new_spec)
        system_particles = new_spec.system.getNumParticles()
        nb_particles = _get_nb_force(new_spec.system).getNumParticles()
        gb_particles = _get_customgb_force(new_spec.system).getNumParticles()

        self.assertEqual(system_particles, EXPECTED)
        self.assertEqual(nb_particles, EXPECTED)
        self.assertEqual(gb_particles, EXPECTED)

    def test_should_add_residues(self):
        n_orig_residues = self.spec.topology.getNumResidues()
        EXPECTED = n_orig_residues + 2

        new_spec, _ = add_rdc_alignment(self.spec)
        new_spec, _ = add_rdc_alignment(new_spec)
        n_residues = new_spec.topology.getNumResidues()

        self.assertEqual(n_residues, EXPECTED)

    def test_should_be_last_residue(self):
        n_orig_residues = self.spec.topology.getNumResidues()
        EXPECTED1 = indexing.ResidueIndex(n_orig_residues)
        EXPECTED2 = indexing.ResidueIndex(n_orig_residues + 1)

        spec, res_index1 = add_rdc_alignment(self.spec)
        _, res_index2 = add_rdc_alignment(spec)

        self.assertEqual(res_index1, EXPECTED1)
        self.assertEqual(res_index2, EXPECTED2)


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
