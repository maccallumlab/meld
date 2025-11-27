#
# Copyright 2025 by Alberto Perez, Imesh Ranaweera
# All rights reserved
#

import unittest
import numpy as np
from openmm import app, unit
from meld.system.builders.grappa import GrappaOptions, GrappaSystemBuilder
from meld.system.builders.spec import SystemSpec

# Imports needed for topology creation
from openmm.app import PDBFile, Modeller, ForceField
from io import StringIO
import textwrap
import tempfile
import os

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
        Build a simple dipeptide topology using Modeller + ForceField,
        exactly mirroring the logic used in the working MELD setup.py.
        """
        
        # 1. Create temporary sequence.dat (required by MELD)
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.sequence_dat_path = os.path.join(cls.temp_dir.name, 'sequence.dat')
        with open(cls.sequence_dat_path, 'w') as f:
            # Assuming a dipeptide, the sequence must match the PDB
            f.write("ALA ALA\n")

        # 2. Minimal ALA-ALA PDB (N-terminal and C-terminal capped)
        pdb_text = textwrap.dedent("""
        ATOM      1  N   ALA A   1       -0.000   1.458   0.000  1.00  0.00                    N
        ATOM      2  H1  ALA A   1        0.000   2.090   0.800  1.00  0.00                    H
        ATOM      3  H2  ALA A   1       -0.100   1.200  -0.900  1.00  0.00                    H
        ATOM      4  H3  ALA A   1       -0.800   1.100   0.500  1.00  0.00                    H
        ATOM      5  CA  ALA A   1        1.214   0.807   0.000  1.00  0.00                    C
        ATOM      6  HA  ALA A   1        1.200   0.050  -0.700  1.00  0.00                    H
        ATOM      7  CB  ALA A   1        1.200  -0.700   0.400  1.00  0.00                    C
        ATOM      8  HB1 ALA A   1        2.200  -1.000   0.200  1.00  0.00                    H
        ATOM      9  HB2 ALA A   1        0.700  -1.200  -0.400  1.00  0.00                    H
        ATOM     10  HB3 ALA A   1        1.700  -1.200   1.300  1.00  0.00                    H
        ATOM     11  C   ALA A   1        2.400   1.668   0.100  1.00  0.00                    C
        ATOM     12  O   ALA A   1        3.400   1.268   0.600  1.00  0.00                    O
        ATOM     13  N   ALA A   2        2.300   2.900  -0.100  1.00  0.00                    N
        ATOM     14  H   ALA A   2        1.600   3.400  -0.600  1.00  0.00                    H
        ATOM     15  CA  ALA A   2        3.400   3.700  -0.600  1.00  0.00                    C
        ATOM     16  HA  ALA A   2        3.100   4.600  -0.900  1.00  0.00                    H
        ATOM     17  CB  ALA A   2        3.100   4.100  -2.000  1.00  0.00                    C
        ATOM     18  HB1 ALA A   2        2.100   4.500  -2.100  1.00  0.00                    H
        ATOM     19  HB2 ALA A   2        3.800   4.900  -2.200  1.00  0.00                    H
        ATOM     20  HB3 ALA A   2        3.400   3.200  -2.700  1.00  0.00                    H
        ATOM     21  C   ALA A   2        4.700   3.100  -0.200  1.00  0.00                    C
        ATOM     22  O   ALA A   2        5.700   3.500   0.200  1.00  0.00                    O
        ATOM     23  OXT ALA A   2        4.700   4.200  -0.800  1.00  0.00                    O
        TER
        END
        """).lstrip()
        
        # 3. Save the PDB text to a temporary file, mimicking protein_min.pdb
        cls.pdb_file_path = os.path.join(cls.temp_dir.name, 'protein_min.pdb')
        with open(cls.pdb_file_path, 'w') as f:
            f.write(pdb_text)

        # 4. Topology Generation
        pdb_file = PDBFile(cls.pdb_file_path)
        
        # Use the exact force field files 
        forcefield = ForceField('amber14/protein.ff14SB.xml', 'implicit/gbn2.xml') 
        modeller = Modeller(pdb_file.topology, pdb_file.positions)
        modeller.addHydrogens(forcefield) # This generates the final, parameterized topology

        cls.topology = modeller.topology
        cls.positions = modeller.positions
        
        cls.expected_atom_count = 23


    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        cls.temp_dir.cleanup()

    # ------------------------------------------------------------------
    @unittest.skipUnless(GRAPPA_INSTALLED, "grappa-ff not installed.")
    def test_default_build(self):
        """Test that a system builds with default settings and uses 2.0 fs timestep."""
        
        options = GrappaOptions(
            solvation_type="implicit",
            grappa_model_tag="grappa-1.4.0",
            base_forcefield_files=['amber14/protein.ff14SB.xml', 'implicit/gbn2.xml'], 
            use_big_timestep=False,
            use_bigger_timestep=False,
        )
        builder = GrappaSystemBuilder(options)
        
        # The build process must succeed
        spec = builder.build_system(self.topology, self.positions)

        self.assertIsInstance(spec, SystemSpec)
        
        # 1. Atom Count Check
        self.assertEqual(spec.topology.getNumAtoms(), self.expected_atom_count)
        self.assertEqual(spec.system.getNumParticles(), self.expected_atom_count)
        self.assertEqual(spec.coordinates.shape[0], self.expected_atom_count)

        # 2. Integrator Checks 
        integ = spec.integrator
        
        # Check type (should be LangevinIntegrator)
        self.assertIsInstance(integ, unit.openmm.LangevinIntegrator)
        
        # Check Temperature
        self.assertAlmostEqual(
            integ.getTemperature().value_in_unit(unit.kelvin), 
            options.default_temperature.value_in_unit(unit.kelvin), 
            delta=0.1
        )
        
        # Check Timestep (must be 2.0 fs)
        self.assertAlmostEqual(
            integ.getStepSize().value_in_unit(unit.femtoseconds), 
            2.0, 
            delta=1e-3
        )
        
        # 3. System Finalization (Optional, but good practice)
        # Try to calculate energy to ensure the Grappa force is applied and works
        try:
            context = unit.openmm.Context(spec.system, integ)
            context.setPositions(spec.coordinates * unit.nanometers)
            state = context.getState(getEnergy=True)
            energy = state.getPotentialEnergy()
            self.assertIsInstance(energy, unit.Quantity)
            context = None # clean up
        except Exception as e:
            self.fail(f"Failed to create Context or calculate energy after Grappa parametrization: {e}")


if __name__ == "__main__":
    unittest.main()