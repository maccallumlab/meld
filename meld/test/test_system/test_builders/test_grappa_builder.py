#
# Copyright 2025 by Alberto Perez, Imesh Ranaweera
# All rights reserved
#

import unittest
import os
import textwrap
import numpy as np
import io

# Import necessary OpenMM components
from openmm import app, unit as u
from openmm.app import PDBFile, Modeller, ForceField

# Import MELD components
from meld.system.builders.grappa.options import GrappaOptions
from meld.system.builders.grappa.builder import GrappaSystemBuilder 


# Try importing grappa-ff
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
        Build a simple ALA-ALA topology using the working Modeller + ForceField method 
        (identical to the successful MELD setup.py).
        """

        # Minimal ALA-ALA PDB (N-terminal NH3+, C-terminal COO-) - 23 atoms total
        # Using f-string formatting for reliable, fixed-width PDB lines.
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
                f"ATOM  {serial:5d} {name:^4s}{resname:>4s} {chain:1s}{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2s}\n"
            )
            pdb_lines.append(line)
        pdb_lines.append('TER\n')
        pdb_lines.append('END\n')

        pdb_text = ''.join(pdb_lines)

        # 1. Load PDB from string
        pdb = PDBFile(io.StringIO(pdb_text))
        
        # 2. Use the exact force field files from the working MELD setup.py
        forcefield = ForceField('amber14/protein.ff14SB.xml', 'implicit/gbn2.xml') 
        modeller = Modeller(pdb.topology, pdb.positions)
        
        # 3. Add Hydrogens (Applies force field templates and topology)
        modeller.addHydrogens(forcefield) 

        cls.topology = modeller.topology
        cls.positions = modeller.positions
        cls.expected_atom_count = 23

    def test_build_system(self):
        """
        Tests the Grappa system building process, checking atom count and 2.0 fs timestep.
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

        # 1. Atom Count Check
        self.assertEqual(system.getNumParticles(), self.expected_atom_count, 
                         f"System particle count mismatch. Expected {self.expected_atom_count}, got {system.getNumParticles()}.")

        # 2. Timestep Check (CRITICAL: Must be 2.0 fs)
        integ = spec.integrator
        self.assertAlmostEqual(
            integ.getStepSize().value_in_unit(u.femtoseconds), 
            2.0, 
            delta=1e-3,
            msg="Integrator timestep is not 2.0 fs."
        )

        # 3. Check for Grappa Force Term (Ensuring the model was successfully applied)
        grappa_force_found = False
        for i in range(system.getNumForces()):
            force = system.getForce(i)
            # Check for the unique custom force added by Grappa
            if type(force).__name__ == 'CustomNonbondedForce':
                if 'grappa_nonbonded' in force.getName().lower():
                    grappa_force_found = True
                    break
            
        self.assertTrue(grappa_force_found, "Grappa force term was not found in the OpenMM system.")


if __name__ == '__main__':
    unittest.main()