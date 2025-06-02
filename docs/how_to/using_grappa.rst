.. _using_grappa:

Using the Grappa Force Field
============================

MELD supports the integration of the machine-learned force field Grappa for predicting bonded parameters. Grappa works in conjunction with a classical force field, which provides the nonbonded parameters (e.g., charges, Lennard-Jones).

Installation
------------

To use the Grappa force field with MELD, you need to install the ``grappa-ff`` package:

.. code-block:: bash

    pip install grappa-ff

Ensure that ``grappa-ff`` and its dependencies (like PyTorch and DGL) are compatible with your environment. For inference (which is what MELD does), a CPU version is usually sufficient.

Using Grappa in MELD
--------------------

To set up a system with Grappa, you will use the ``GrappaOptions`` and ``GrappaSystemBuilder`` classes from MELD.

1.  **Import necessary classes:**

    .. code-block:: python

        from meld.system.builders.grappa import GrappaOptions, GrappaSystemBuilder
        from openmm.app import PDBFile, Topology
        from openmm import unit

2.  **Prepare your molecular system:**
    You'll need an OpenMM ``Topology`` and the initial ``positions``. These can often be loaded from a PDB file.

    .. code-block:: python

        pdb = PDBFile('your_molecule.pdb')
        topology = pdb.topology
        positions = pdb.positions
        # Optional: define box vectors if you have a periodic system
        # box_vectors = pdb.topology.getPeriodicBoxVectors()

3.  **Configure GrappaOptions:**
    Specify the Grappa model tag and the base force field files. Other options like temperature, cutoff, and timestep settings can also be configured.

    .. code-block:: python

        grappa_options = GrappaOptions(
            grappa_model_tag='grappa-1.4',  # Or any other valid Grappa model tag
            base_forcefield_files=['amber/ff14SB.xml', 'amber/tip3p.xml'], # For protein systems
            default_temperature=300.0 * unit.kelvin,
            cutoff=1.0 * unit.nanometer,  # For PME, or None for NoCutoff
            use_big_timestep=False # Set to True for 3fs timestep with HMR
        )

4.  **Create the GrappaSystemBuilder and build the system:**

    .. code-block:: python

        builder = GrappaSystemBuilder(grappa_options)
        # If you have box vectors:
        # system_spec = builder.build_system(topology, positions, box_vectors=box_vectors)
        # If no explicit box vectors (e.g. implicit solvent or vacuum):
        system_spec = builder.build_system(topology, positions)

        # The system_spec object now contains your OpenMM system, topology, integrator, etc.
        # You can then pass this to the MELD System object:
        # from meld.system import System
        # meld_system = System(system_spec, ...) # Further MELD setup

How it Works
------------

The ``GrappaSystemBuilder`` first sets up an OpenMM ``System`` using the provided base force field files (e.g., AMBER ff14SB). This step defines the nonbonded interactions and initial bonded parameters.
Then, it utilizes the specified Grappa model (e.g., ``grappa-1.4``) to predict and replace the bonded parameters (bonds, angles, torsions) in the OpenMM ``System``. The nonbonded parameters from the base force field remain unchanged.

Example MELD Setup Snippet
--------------------------

Here's a more complete snippet showing how ``GrappaSystemBuilder`` fits into a MELD setup script:

.. code-block:: python

    from meld.system import System, SystemDirector, RunOptions
    from meld.system.restraints import RestraintGroup, SelectivelyLabeledRestraintScaler
    from meld.system.temperature import TemperatureScaler
    from meld.system.builders.grappa import GrappaOptions, GrappaSystemBuilder
    from meld.comm import Communicator
    from meld.store import Store

    from openmm.app import PDBFile
    from openmm import unit

    def setup_meld_system():
        # Load topology and positions
        pdb = PDBFile('ala_dipeptide.pdb') # Replace with your PDB
        topology = pdb.topology
        positions = pdb.positions
        # box_vectors = pdb.topology.getPeriodicBoxVectors() # If applicable

        # Configure Grappa
        grappa_opts = GrappaOptions(
            grappa_model_tag='latest', # Use a specific tag like 'grappa-1.4' for reproducibility
            base_forcefield_files=['amber/ff14SB.xml', 'amber/tip3p.xml'],
            default_temperature=300.0 * unit.kelvin,
            cutoff=1.0 * unit.nanometer # e.g. for PME for explicit solvent simulations
                                       # or None for implicit/vacuum
        )
        grappa_builder = GrappaSystemBuilder(grappa_opts)

        # Build the system specification
        system_spec = grappa_builder.build_system(topology, positions) # Add box_vectors if needed

        # MELD System object
        meld_system = System(
            system_spec=system_spec,
            communicator=Communicator(), # Replace with actual communicator if using MPI
            restraints=None, # Add your restraint groups here
            director=SystemDirector(), # Customize director if needed
            options=RunOptions() # Customize run options if needed
        )
        return meld_system

    # To run this (simplified):
    # if __name__ == '__main__':
    #     store = Store(0, giám đốc=None, backup_filename='backup.dat', mode='w')
    #     system = setup_meld_system()
    #     # Further simulation setup...

Remember to consult the official Grappa documentation for details on available models and their capabilities.
