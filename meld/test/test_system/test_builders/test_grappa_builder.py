#
# Copyright 2023 The MELD Contributors
# All rights reserved
#

import unittest
import os
import numpy as np
from openmm import app, unit
from meld.system.builders.grappa import GrappaOptions, GrappaSystemBuilder
from meld.system.builders.spec import SystemSpec

# Attempt to import grappa to check if it's installed
try:
    import grappa
    GRAPPA_INSTALLED = True
except ImportError:
    GRAPPA_INSTALLED = False

# Minimal PDB content for alanine dipeptide (ACE-ALA-NME)
ALA_DIPEPTIDE_PDB_CONTENT = """
ATOM      1  CH3 ACE     1      -1. ACE     1      -2.531  -0.425  -0.087  1.00  0.00           C
ATOM      2  H1  ACE     1      -1. ACE     1      -2.886  -1.460  -0.087  1.00  0.00           H
ATOM      3  H2  ACE     1      -1. ACE     1      -2.886   0.030   0.843  1.00  0.00           H
ATOM      4  H3  ACE     1      -1. ACE     1      -2.886   0.030  -0.947  1.00  0.00           H
ATOM      5  C   ACE     1      -1. ACE     1      -1.004  -0.307  -0.004  1.00  0.00           C
ATOM      6  O   ACE     1      -1. ACE     1      -0.490  -1.301   0.420  1.00  0.00           O
ATOM      7  N   ALA     2       0. ALA     2      -0.300   0.826  -0.387  1.00  0.00           N
ATOM      8  H   ALA     2       0. ALA     2       0.709   0.751  -0.071  1.00  0.00           H
ATOM      9  CA  ALA     2       0. ALA     2      -0.848   2.039   0.170  1.00  0.00           C
ATOM     10  HA  ALA     2       0. ALA     2      -0.271   2.835  -0.279  1.00  0.00           H
ATOM     11  CB  ALA     2       0. ALA     2      -0.747   2.390   1.660  1.00  0.00           C
ATOM     12  HB1 ALA     2       0. ALA     2      -1.208   1.560   2.190  1.00  0.00           H
ATOM     13  HB2 ALA     2       0. ALA     2       0.290   2.548   1.974  1.00  0.00           H
ATOM     14  HB3 ALA     2       0. ALA     2      -1.264   3.298   1.870  1.00  0.00           H
ATOM     15  C   ALA     2       0. ALA     2      -2.257   1.897  -0.394  1.00  0.00           C
ATOM     16  O   ALA     2       0. ALA     2      -2.632   2.699  -1.210  1.00  0.00           O
ATOM     17  N   NME     3       1. NME     3      -3.009   0.848   0.143  1.00  0.00           N
ATOM     18  H   NME     3       1. NME     3      -2.619   0.153   0.713  1.00  0.00           H
ATOM     19  CH3 NME     3       1. NME     3      -4.413   0.768  -0.269  1.00  0.00           C
ATOM     20  H1  NME     3       1. NME     3      -4.799   1.799  -0.269  1.00  0.00           H
ATOM     21  H2  NME     3       1. NME     3      -4.799   0.298   0.641  1.00  0.00           H
ATOM     22  H3  NME     3       1. NME     3      -4.799   0.298  -1.129  1.00  0.00           H
END
"""

# Define a directory for test files
TEST_FILES_DIR = "test_grappa_files"
PDB_FILENAME = os.path.join(TEST_FILES_DIR, "ala_dipeptide.pdb")

# Standard AMBER force fields for base FF
# Using ff14SB for protein and tip3p for water (though no water here)
# These are commonly available with OpenMM installations.
# If these are not available, users might need to ensure their OpenMM is set up with standard FFs.
DEFAULT_BASE_FF = ["amber/ff14SB.xml", "amber/tip3p.xml"]


def ensure_test_pdb_exists():
    if not os.path.exists(TEST_FILES_DIR):
        os.makedirs(TEST_FILES_DIR)
    if not os.path.exists(PDB_FILENAME):
        with open(PDB_FILENAME, "w") as f:
            f.write(ALA_DIPEPTIDE_PDB_CONTENT)
    # Check if default base FF files exist, warn if not
    # This is a soft check; OpenMM will error if they truly don't exist
    ff = app.ForceField() # Test instantiation
    try:
        # Attempt to load to see if they are in OpenMM's search path
        app.ForceField(*DEFAULT_BASE_FF)
    except Exception as e:
        print(f"Warning: Default base force field files ({DEFAULT_BASE_FF}) might not be accessible by OpenMM: {e}")
        print("Tests requiring these files might fail if OpenMM cannot find them.")


class TestGrappaSystemBuilder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ensure_test_pdb_exists()
        cls.pdb = app.PDBFile(PDB_FILENAME)
        cls.topology = cls.pdb.topology
        cls.positions = cls.pdb.positions

    @unittest.skipUnless(GRAPPA_INSTALLED, "Grappa-ff is not installed.")
    def test_build_system_default_options(self):
        """Test building a system with default GrappaOptions."""
        options = GrappaOptions(base_forcefield_files=DEFAULT_BASE_FF) # Use readily available FFs
        builder = GrappaSystemBuilder(options)

        spec = builder.build_system(self.topology, self.positions)

        self.assertIsInstance(spec, SystemSpec)
        self.assertIsNotNone(spec.system)
        self.assertIsNotNone(spec.topology)
        self.assertIsNotNone(spec.integrator)
        self.assertEqual(spec.topology.getNumAtoms(), self.topology.getNumAtoms())
        self.assertIsInstance(spec.integrator, app.LangevinIntegrator)
        # Check default temperature
        self.assertAlmostEqual(spec.integrator.getTemperature().value_in_unit(unit.kelvin), 300.0, delta=0.01)
        # Check default timestep (2fs)
        self.assertAlmostEqual(spec.integrator.getStepSize().value_in_unit(unit.femtoseconds), 2.0, delta=0.01)


    @unittest.skipUnless(GRAPPA_INSTALLED, "Grappa-ff is not installed.")
    def test_build_system_with_cutoff(self):
        """Test building a system with a nonbonded cutoff."""
        cutoff_val = 1.0  # nm
        options = GrappaOptions(cutoff=cutoff_val, base_forcefield_files=DEFAULT_BASE_FF)
        builder = GrappaSystemBuilder(options)
        spec = builder.build_system(self.topology, self.positions)

        self.assertIsInstance(spec, SystemSpec)
        nonbonded_force = None
        for force in spec.system.getForces():
            if isinstance(force, app.NonbondedForce):
                nonbonded_force = force
                break
        self.assertIsNotNone(nonbonded_force)
        self.assertEqual(nonbonded_force.getNonbondedMethod(), app.NonbondedForce.PME)
        self.assertAlmostEqual(nonbonded_force.getCutoffDistance().value_in_unit(unit.nanometer), cutoff_val, delta=0.01)

    @unittest.skipUnless(GRAPPA_INSTALLED, "Grappa-ff is not installed.")
    def test_build_system_big_timestep(self):
        """Test building a system with use_big_timestep=True."""
        options = GrappaOptions(use_big_timestep=True, base_forcefield_files=DEFAULT_BASE_FF)
        builder = GrappaSystemBuilder(options)
        spec = builder.build_system(self.topology, self.positions)

        self.assertIsInstance(spec, SystemSpec)
        self.assertIsInstance(spec.integrator, app.LangevinIntegrator)
        self.assertAlmostEqual(spec.integrator.getStepSize().value_in_unit(unit.femtoseconds), 3.0, delta=0.01)
        
        # Check for hydrogen mass > 1
        found_heavy_h = False
        for atom in spec.topology.atoms():
            if atom.element == app.Element.getBySymbol('H'):
                mass = spec.system.getAtomMass(atom.index)
                if mass.value_in_unit(unit.dalton) > 1.5: # Default is ~1, heavy H is 3 or 4
                    found_heavy_h = True
                    break
        self.assertTrue(found_heavy_h, "Hydrogen mass was not increased with use_big_timestep.")


    @unittest.skipUnless(GRAPPA_INSTALLED, "Grappa-ff is not installed.")
    def test_build_system_bigger_timestep(self):
        """Test building a system with use_bigger_timestep=True."""
        options = GrappaOptions(use_bigger_timestep=True, base_forcefield_files=DEFAULT_BASE_FF)
        builder = GrappaSystemBuilder(options)
        spec = builder.build_system(self.topology, self.positions)

        self.assertIsInstance(spec, SystemSpec)
        self.assertIsInstance(spec.integrator, app.LangevinIntegrator)
        self.assertAlmostEqual(spec.integrator.getStepSize().value_in_unit(unit.femtoseconds), 4.0, delta=0.01)

        # Check for hydrogen mass > 1
        found_heavy_h = False
        for atom in spec.topology.atoms():
            if atom.element == app.Element.getBySymbol('H'):
                mass = spec.system.getAtomMass(atom.index)
                if mass.value_in_unit(unit.dalton) > 1.5: # Default is ~1, heavy H is 3 or 4
                    found_heavy_h = True
                    break
        self.assertTrue(found_heavy_h, "Hydrogen mass was not increased with use_bigger_timestep.")


    def test_grappa_not_installed_message(self):
        """Test that a message is printed if grappa is not installed."""
        if not GRAPPA_INSTALLED:
            # This test implicitly passes if the skipUnless decorator works.
            # We can add a print statement to make it explicit during testing.
            print("Grappa-ff not installed, GrappaSystemBuilder tests are being skipped.")
        else:
            self.skipTest("Grappa-ff is installed, so this test is not applicable.")
            
    @unittest.skipUnless(GRAPPA_INSTALLED, "Grappa-ff is not installed.")
    def test_invalid_grappa_model_tag(self):
        """Test that a RuntimeError is raised for an invalid Grappa model tag."""
        options = GrappaOptions(grappa_model_tag="invalid-tag-that-does-not-exist", base_forcefield_files=DEFAULT_BASE_FF)
        builder = GrappaSystemBuilder(options)
        with self.assertRaisesRegex(RuntimeError, "Grappa model loading failed"):
            builder.build_system(self.topology, self.positions)


if __name__ == "__main__":
    ensure_test_pdb_exists() # Ensure PDB is there before running if module is run directly
    unittest.main()
