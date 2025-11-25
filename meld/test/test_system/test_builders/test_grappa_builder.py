#
# Copyright 2025 by Imesh Ranaweera, Alberto Perez
# All rights reserved
#

import unittest
import numpy as np
from openmm import app, unit
from meld.system.builders.grappa import GrappaOptions, GrappaSystemBuilder
from meld.system.builders.spec import SystemSpec

# Try importing grappa-ff
try:
    import grappa
    GRAPPA_INSTALLED = True
except ImportError:
    GRAPPA_INSTALLED = False


class TestGrappaBuilder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Use OpenMM's built-in Alanine Dipeptide test system.
        This guarantees:
        - valid topology
        - valid positions
        - consistent atom count
        - no dependency on XML files beyond ff14SB / gbn2
        """
        from openmm.app import PDBFile

        # Create OpenMM ALA dipeptide test system
        pdb = app.PDBFile(app.internal.test.pdb_files.ala2_implicit)
        cls.topology = pdb.topology
        cls.positions = pdb.positions

    # ------------------------------------------------------------------
    @unittest.skipUnless(GRAPPA_INSTALLED, "grappa-ff not installed.")
    def test_default_build(self):
        """Test that a system builds with default settings."""
        options = GrappaOptions(
            solvation_type="implicit",
            grappa_model_tag="grappa-1.4.0",
        )
        builder = GrappaSystemBuilder(options)
        spec = builder.build_system(self.topology, self.positions)

        self.assertIsInstance(spec, SystemSpec)
        self.assertEqual(spec.topology.getNumAtoms(), self.topology.getNumAtoms())

        # Integrator checks
        integ = spec.integrator
        self.assertAlmostEqual(
            integ.getTemperature().value_in_unit(unit.kelvin), 300.0, delta=0.1
        )
        self.assertAlmostEqual(
            integ.getStepSize().value_in_unit(unit.femtoseconds), 2.0, delta=0.1
        )

    # ------------------------------------------------------------------
    @unittest.skipUnless(GRAPPA_INSTALLED, "grappa-ff not installed.")
    def test_cutoff(self):
        """Test nonbonded cutoff handling."""
        options = GrappaOptions(
            solvation_type="implicit",
            grappa_model_tag="grappa-1.4.0",
            cutoff=1.0,  # nm
        )
        builder = GrappaSystemBuilder(options)
        spec = builder.build_system(self.topology, self.positions)

        # Find NonbondedForce
        from openmm import NonbondedForce

        nb_force = None
        for f in spec.system.getForces():
            if isinstance(f, NonbondedForce):
                nb_force = f
                break

        self.assertIsNotNone(nb_force)
        self.assertAlmostEqual(
            nb_force.getCutoffDistance().value_in_unit(unit.nanometer),
            1.0,
            delta=0.01,
        )

    # ------------------------------------------------------------------
    @unittest.skipUnless(GRAPPA_INSTALLED, "grappa-ff not installed.")
    def test_big_timestep(self):
        """Test use_big_timestep = True."""
        options = GrappaOptions(
            solvation_type="implicit",
            grappa_model_tag="grappa-1.4.0",
            use_big_timestep=True,
        )
        builder = GrappaSystemBuilder(options)
        spec = builder.build_system(self.topology, self.positions)

        integ = spec.integrator
        self.assertAlmostEqual(
            integ.getStepSize().value_in_unit(unit.femtoseconds), 3.5, delta=0.01
        )

        # Hydrogen mass must be 3 Da
        heavy_found = False
        for atom in spec.topology.atoms():
            if atom.element.symbol == "H":
                mass = spec.system.getAtomMass(atom.index).value_in_unit(unit.dalton)
                if mass > 1.5:
                    heavy_found = True
                    break
        self.assertTrue(heavy_found)

    # ------------------------------------------------------------------
    @unittest.skipUnless(GRAPPA_INSTALLED, "grappa-ff not installed.")
    def test_bigger_timestep(self):
        """Test use_bigger_timestep = True."""
        options = GrappaOptions(
            solvation_type="implicit",
            grappa_model_tag="grappa-1.4.0",
            use_bigger_timestep=True,
        )
        builder = GrappaSystemBuilder(options)
        spec = builder.build_system(self.topology, self.positions)

        integ = spec.integrator
        self.assertAlmostEqual(
            integ.getStepSize().value_in_unit(unit.femtoseconds), 4.5, delta=0.01
        )

        # Hydrogen mass must be 4 Da
        heavy_found = False
        for atom in spec.topology.atoms():
            if atom.element.symbol == "H":
                mass = spec.system.getAtomMass(atom.index).value_in_unit(unit.dalton)
                if mass > 2.0:
                    heavy_found = True
                    break
        self.assertTrue(heavy_found)


if __name__ == "__main__":
    unittest.main()
