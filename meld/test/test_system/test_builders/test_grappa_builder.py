#
# Copyright 2025 by Alberto Perez, Imesh Ranaweera
# All rights reserved
#

import unittest
import io
import numpy as np

from openmm import app, unit as u
from openmm.app import PDBFile, Modeller, ForceField

from meld.system.builders.grappa.options import GrappaOptions
from meld.system.builders.grappa.builder import GrappaSystemBuilder


# Check if grappa is installed
try:
    import grappa
    GRAPPA_INSTALLED = True
except ImportError:
    GRAPPA_INSTALLED = False
    print("Warning: Grappa not installed. Skipping tests.")


@unittest.skipUnless(GRAPPA_INSTALLED, "grappa-ff not installed.")
class TestGrappaBuilder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Build a simple ALA-ALA topology using Modeller + ff14SB + GBN2,
        and add hydrogens exactly like MELD does.
        """

        # Minimal ALA-ALA PDB (23 atoms before hydrogen completion)
        atoms = [
            (1, 'N',  'ALA', 'A', 1, -0.000,  1.458,  0.000, 'N'),
            (2, 'H1', 'ALA', 'A', 1,  0.000,  2.090,  0.800, 'H'),
            (3, 'H2', 'ALA', 'A', 1, -0.100,  1.200, -0.900, 'H'),
            (4, 'H3', 'ALA', 'A', 1, -0.800,  1.100,  0.500, 'H'),
            (5, 'CA', 'ALA', 'A', 1,  1.214,  0.807,  0.000, 'C'),
            (6, 'HA', 'ALA', 'A', 1,  1.200,  0.050, -0.700, 'H'),
            (7, 'CB', 'ALA', 'A', 1,  1.200, -0.700,  0.400, 'C'),
            (8, 'HB1','ALA', 'A', 1,  2.200, -1.000,  0.200, 'H'),
            (9, 'HB2','ALA', 'A', 1,  0.700, -1.200, -0.400, 'H'),
            (10,'HB3','ALA', 'A', 1,  1.700, -1.200,  1.300, 'H'),
            (11,'C',  'ALA', 'A', 1,  2.400,  1.668,  0.100, 'C'),
            (12,'O',  'ALA', 'A', 1,  3.400,  1.268,  0.600, 'O'),
            (13,'N',  'ALA', 'A', 2,  2.300,  2.900, -0.100, 'N'),
            (14,'H',  'ALA', 'A', 2,  1.600,  3.400, -0.600, 'H'),
            (15,'CA', 'ALA', 'A', 2,  3.400,  3.700, -0.600, 'C'),
            (16,'HA', 'ALA', 'A', 2,  3.100,  4.600, -0.900, 'H'),
            (17,'CB', 'ALA', 'A', 2,  3.100,  4.100, -2.000, 'C'),
            (18,'HB1','ALA', 'A', 2,  2.100,  4.500, -2.100, 'H'),
            (19,'HB2','ALA', 'A', 2,  3.800,  4.900, -2.200, 'H'),
            (20,'HB3','ALA', 'A', 2,  3.400,  3.200, -2.700, 'H'),
            (21,'C',  'ALA', 'A', 2,  4.700,  3.100, -0.200, 'C'),
            (22,'O',  'ALA', 'A', 2,  5.700,  3.500,  0.200, 'O'),
            (23,'OXT','ALA', 'A', 2,  4.700,  4.200, -0.800, 'O'),
        ]

        pdb_lines = []
        for serial, name, resname, chain, resseq, x, y, z, element in atoms:
            line = (
                f"ATOM  {serial:5d} {name:^4s}{resname:>4s} {chain}{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2s}\n"
            )
            pdb_lines.append(line)
        pdb_lines.append("TER\nEND\n")

        pdb = PDBFile(io.StringIO("".join(pdb_lines)))

        forcefield = ForceField('amber14/protein.ff14SB.xml', 'implicit/gbn2.xml')
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield)

        cls.topology = modeller.topology
        cls.positions = modeller.positions

        cls.expected_atom_count = len(list(cls.topology.atoms()))  # ✔ FIXED

    def test_build_system(self):
        """
        Validate atom count, timestep, and that Grappa forces were added.
        """

        grappa_options = GrappaOptions(
            solvation_type="implicit",
            grappa_model_tag="grappa-1.4.0",
            base_forcefield_files=['amber14/protein.ff14SB.xml', 'implicit/gbn2.xml'],
            use_big_timestep=False,
            use_bigger_timestep=False,
        )

        builder = GrappaSystemBuilder(grappa_options)
        spec = builder.build_system(self.topology, self.positions)
        system = spec.system

        # 1. Atom count
        self.assertEqual(
            system.getNumParticles(),
            self.expected_atom_count,
            f"Expected {self.expected_atom_count} atoms, got {system.getNumParticles()}."
        )

        # 2. Timestep must be 2.0 fs
        dt = spec.integrator.getStepSize().value_in_unit(u.femtoseconds)
        self.assertAlmostEqual(dt, 2.0, delta=1e-3, msg="Integrator timestep is not 2.0 fs.")

        # 3. Grappa force presence check (name-based, robust)
        grappa_found = any(
            "grappa" in system.getForce(i).getName().lower()
            for i in range(system.getNumForces())
        )

        self.assertTrue(
            grappa_found,
            "No Grappa forces detected in the system (force.name does not contain 'grappa')."
        )


if __name__ == '__main__':
    unittest.main()
