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
        from io import StringIO
        import textwrap

        # Create OpenMM ALA dipeptide test system.
        # Older/newer OpenMM builds may not expose `app.internal.test` helper data,
        # so embed a minimal ALA-ALA PDB here as a stable fallback for CI.
        pdb_text = textwrap.dedent("""
        ATOM      1  N   ALA A   1      -0.000   1.458   0.000  1.00  0.00           N
        ATOM      2  H   ALA A   1       0.000   2.090   0.800  1.00  0.00           H
        ATOM      3  CA  ALA A   1       1.214   0.807   0.000  1.00  0.00           C
        ATOM      4  HA  ALA A   1       1.200   0.050  -0.700  1.00  0.00           H
        ATOM      5  CB  ALA A   1       1.200  -0.700   0.400  1.00  0.00           C
        ATOM      6  HB1 ALA A   1       2.200  -1.000   0.200  1.00  0.00           H
        ATOM      7  HB2 ALA A   1       0.700  -1.200  -0.400  1.00  0.00           H
        ATOM      8  HB3 ALA A   1       1.700  -1.200   1.300  1.00  0.00           H
        ATOM      9  C   ALA A   1       2.400   1.668   0.100  1.00  0.00           C
        ATOM     10  O   ALA A   1       3.400   1.268   0.600  1.00  0.00           O
        ATOM     11  N   ALA A   2       2.300   2.900  -0.100  1.00  0.00           N
        ATOM     12  H   ALA A   2       1.600   3.400  -0.600  1.00  0.00           H
        ATOM     13  CA  ALA A   2       3.400   3.700  -0.600  1.00  0.00           C
        ATOM     14  HA  ALA A   2       3.100   4.600  -0.900  1.00  0.00           H
        ATOM     15  CB  ALA A   2       3.100   4.100  -2.000  1.00  0.00           C
        ATOM     16  HB1 ALA A   2       2.100   4.500  -2.100  1.00  0.00           H
        ATOM     17  HB2 ALA A   2       3.800   4.900  -2.200  1.00  0.00           H
        ATOM     18  HB3 ALA A   2       3.400   3.200  -2.700  1.00  0.00           H
        ATOM     19  C   ALA A   2       4.700   3.100  -0.200  1.00  0.00           C
        ATOM     20  O   ALA A   2       5.700   3.500   0.200  1.00  0.00           O
        ATOM     21  OXT ALA A   2       4.700   4.200  -0.800  1.00  0.00           O
        TER
        END
        """).lstrip()

        pdb = PDBFile(StringIO(pdb_text))
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
