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
# Note: Assuming builder.py and options.py are correctly placed for this import to work
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
            # CRITICAL FIX: Ensure coordinates are in columns 31-38 (X), 39-46 (Y), 47-54 (Z).
            pdb_text = textwrap.dedent("""
    ATOM      1  N   ALA A   1     -0.000  1.458  0.000  1.00  0.00                    N
    ATOM      2 H1  ALA A   1      0.000  2.090  0.800  1.00  0.00                    H
    ATOM      3 H2  ALA A   1     -0.100  1.200 -0.900  1.00  0.00                    H
    ATOM      4 H3  ALA A   1     -0.800  1.100  0.500  1.00  0.00                    H
    ATOM      5 CA  ALA A   1      1.214  0.807  0.000  1.00  0.00                    C
    ATOM      6 HA  ALA A   1      1.200  0.050 -0.700  1.00  0.00                    H
    ATOM      7 CB  ALA A   1      1.200 -0.700  0.400  1.00  0.00                    C
    ATOM      8 HB1 ALA A   1      2.200 -1.000  0.200  1.00  0.00                    H
    ATOM      9 HB2 ALA A   1      0.700 -1.200 -0.400  1.00  0.00                    H
    ATOM     10 HB3 ALA A   1      1.700 -1.200  1.300  1.00  0.00                    H
    ATOM     11 C   ALA A   1      2.400  1.668  0.100  1.00  0.00                    C
    ATOM     12 O   ALA A   1      3.400  1.268  0.600  1.00  0.00                    O
    ATOM     13 N   ALA A   2      2.300  2.900 -0.100  1.00  0.00                    N
    ATOM     14 H   ALA A   2      1.600  3.400 -0.600  1.00  0.00                    H
    ATOM     15 CA  ALA A   2      3.400  3.700 -0.600  1.00  0.00                    C
    ATOM     16 HA  ALA A   2      3.100  4.600 -0.900  1.00  0.00                    H
    ATOM     17 CB  ALA A   2      3.100  4.100 -2.000  1.00  0.00                    C
    ATOM     18 HB1 ALA A   2      2.100  4.500 -2.100  1.00  0.00                    H
    ATOM     19 HB2 ALA A   2      3.800  4.900 -2.200  1.00  0.00                    H
    ATOM     20 HB3 ALA A   2      3.400  3.200 -2.700  1.00  0.00                    H
    ATOM     21 C   ALA A   2      4.700  3.100 -0.200  1.00  0.00                    C
    ATOM     22 O   ALA A   2      5.700  3.500  0.200  1.00  0.00                    O
    ATOM     23 OXT ALA A   2      4.700  4.200 -0.800  1.00  0.00                    O
    TER
    END
            """).lstrip()

        # 1. Load PDB from string
        pdb = PDBFile(io.StringIO(pdb_text))
        
        # 2. Use the exact force field files from the working MELD setup.py
        # This tells Modeller how to handle terminal residues and connectivity.
        forcefield = ForceField('amber14/protein.ff14SB.xml', 'implicit/gbn2.xml') 
        modeller = Modeller(pdb.topology, pdb.positions)
        
        # 3. Add Hydrogens (Key step that finalizes the topology for the FF)
        modeller.addHydrogens(forcefield) 

        cls.topology = modeller.topology
        cls.positions = modeller.positions
        cls.expected_atom_count = 23 # The correct count for the ALA-ALA PDB input

    def test_build_system(self):
        """
        Tests the Grappa system building process using the successful ALA-ALA topology.
        Also checks for the critical 2.0 fs timestep.
        """
        
        # Options set to match the default working setup (implicit, 2.0 fs timestep)
        grappa_options = GrappaOptions(
            solvation_type="implicit",
            grappa_model_tag="grappa-1.4.0",
            # Ensure base FF files match setUpClass
            base_forcefield_files=['amber14/protein.ff14SB.xml', 'implicit/gbn2.xml'], 
            use_big_timestep=False,
            use_bigger_timestep=False,
        )
        
        builder = GrappaSystemBuilder(grappa_options)
        spec = builder.build_system(self.topology, self.positions)
        
        # The MELD system object is finalized from the spec
        system = spec.system 

        # 1. Atom Count Check: Ensure the final system matches the expected atom count (23 for ALA-ALA)
        self.assertEqual(system.getNumParticles(), self.expected_atom_count, 
                         f"System particle count mismatch. Expected {self.expected_atom_count}, got {system.getNumParticles()}.")

        # 2. Timestep Check: Ensure the critical 2.0 fs timestep is used
        integ = spec.integrator
        self.assertAlmostEqual(
            integ.getStepSize().value_in_unit(u.femtoseconds), 
            2.0, 
            delta=1e-3,
            msg="Integrator timestep is not 2.0 fs when use_big_timestep/use_bigger_timestep are False."
        )


if __name__ == '__main__':
    # Use unittest.main() as a standard way to run the test
    unittest.main()