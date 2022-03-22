================
Explicit Solvent
================

Building an Explicit Solvent Model
==================================

In MELD, an explicit solvent simulation can be started either from a protein sequence or from 
pre-generated amber topology and coordinate files. As of yet, explicit solvent is not set to 
work from any other input files including PDB.

From Sequence
-------------

A simple explicit solvent model can be built in the setup.py script by first generating a 
protein sequence and setting up an explicit solvent SystemBuilder that defines the system:

.. code-block:: python

   p = subsystem.AmberSubSystemFromSequence("NALA ALA CALA")        
   b = builder.AmberSystemBuilder(solvation="explicit")

The system is then built from the molecular sequence according to the specifications of the 
builder:

.. code-block:: python

        self.system = b.build_system([p])

This set of commands will instruct MELD to use tleap to generate a solvent box with 6 |ang| 
between any atom in the protein and the edge of the box, and produce the necessary amber 
topology (system.top) and coordinate files (system.mdcrd). By default MELD uses the TIP3P 
water model, however ``'spce'``, ``'spceb'``, ``'obc'``, ``'tip4pew'`` are all allowed values 
for ``solvent_forcefield``:

.. code-block:: python

        b = builder.SystemBuilder(explicit_solvent=True, solvent_forcefield="tip3p")

The distance between the protein and the edge of the box (in |Ang|) can also be specified:

.. code-block:: python

        b = builder.SystemBuilder(explicit_solvent=True, solvent_distance=10)

The box is not automatically neutralized, which therefore requires the user to determine the 
charge of the molecule in order to accurately neutralize. Positive and negative ions can be 
added to the water box:

.. code-block:: python

        b = builder.SystemBuilder(explicit_solvent=True, explicit_ions=True, p_ioncount=5, n_ioncount=5)

By default the positive ions are ``Na+``, and the negative ions are ``Cl-``. Allowed values 
for ``p_ions`` include: ``Na+``, ``K+``, ``Li+``, ``Rb+``, ``Cs+``, ``Mg+``. The allowed 
values for `n_ions` include: ``Cl-``, ``I-``, ``Br-``, ``F-``. They can be specified by:

.. code-block:: python

        b = builder.SystemBuilder(explicit_solvent=True, explicit_ions=True, p_ion="K+", p_ioncount=5, n_ion="F-", n_ioncount=5)

As not all forcefields include parameters for all ion types, make sure to select ions that 
are supported by your forcefield.

From Amber Files
----------------

An explicit solvent system can also be generated from pre-made amber topology (.top) and 
coordinate (.crd) files. An example tleap script to generate a 4-mer peptide in a 10 |Ang| 
TIP3P water box is included below::

        xleap
                source leaprc.protein.ff14SB
                foo = {ALA ALA ALA ALA}
                source leaprc.water.tip3p
                solvatebox foo TIP3PBOX 10.0
                saveamberparm foo system.top system.crd
                quit

The system is then built in the setup.py script as:

.. code-block:: python

        s = system.builder.load_amber_system('system.top', 'system.crd')
                                                                                                                                                                                          
When generating the amber files, keep in mind that MELD can only use the following 
specifications:

====================  ===============================================================
Parameter              Allowed Values
====================  ===============================================================
`solvent_forcefield`  `'tip3p'`, `'spce'`, `'spceb'`, `'obc'`, `'tip4ew'`
`p_ions`              `'Na+'`, `'K+'`, `'Li+'`, `'Rb+'`, `'Cs+'`, `'Mg+'`
`n_ions`              `'Cl-'`, `'I-'`, `'Br-'`, `'F-'`
====================  ===============================================================


and is limited to rectangular box shapes.


Setting Options
===============

Once an explicit solvent system is built either with the builder or from pre-generated amber files, it must be declared to the runtime options:

.. code-block:: python

        options = system.RunOptions(solvation='explicit')


Explicit Solvent and Replica Exchange
=====================================

For a simple system, MELD's regular Hamiltonian replica exchange algorithm is likely to be 
sufficient. Keep in mind that, due to the greater size of the system, this will require a 
larger number of replicas and a longer run time, with careful monitoring of replica 
exchanges. The exact parameters are system specific. For larger systems (such as proteins) it 
is advisable to use replica exchange with solute tempering (REST2). A full explanation of the 
method can be found at: 
     
Wang, Lingle, Richard A. Friesner, and B. J. Berne. "Replica exchange with solute 
scaling: a more efficient version of replica exchange with solute tempering (REST2)."
The Journal of Physical Chemistry B 115.30 (2011): 9431-9438.
`DOI : 10.1021/jp204407d <https:// pubs.acs.org/doi/10.1021/jp204407d>`_.

In short, REST2 works by dividing the system into protein and water, and scaling the 
intramolecular potential energy function at a constant temperature. This is physically 
similar to changing the temperature of the protein while keeping the surroundings at the 
target temperature as we climb the temperature ladder. Therefore the acceptance probability 
scales only with the size of the biomolecule, and not with the number of water molecules.

In MELD, we employ REST2 by first declaring a temperature scaler for the target temperature 
of the system:

.. code-block:: python

    s.temperature_scaler = system.ConstantTemperatureScaler(300.)

We then define the REST2 scaler that will be applied to the potential energy function:

.. code-block:: python

   rest2_scaler = system.GeometricTemperatureScaler(alpha_min=0.5, alpha_max=1, temperature_min=300., temperature_max=450.)

This scaler can be geometric or linear. In the above example, the minimum and maximum 
temperatures will be turned into a scaling factor that when applied to the potential energy 
function achieves a physically similar result to scaling the temperature of the replicas from 
``alpha_min`` to ``alpha_max`` across the ladder from ``temperature_min`` to 
``temperature_max``.

In the options, we declare that we are using REST2 by:

.. code-block:: python

        options.use_rest2 = True

And set the REST2 scaler:

.. code-block:: python

        options.rest2_scaler = system.REST2Scaler(300., rest2_scaler)


Periodic Boundary Conditions
============================

Explicit solvent simulations in MELD make use of periodic boundary conditions (PBC’s) through 
OpenMM. In OpenMM, if a periodic box is enforced then the center of every molecule is 
translated so that it lies in the same periodic box. This means that unconnected molecules, 
say a peptide and a protein in a complex that are not bound together, could be translated /
reimaged differently. In OpenMM all atoms involved in each bond are treated as a single 
molecule. Therefore, in MELD we group together all of the atom pairs that are restrained by: 
distance, hyperbolic distance, torsions, Gaussian mixture models (GMM’s), distance profiles, 
and torsion profiles. This creates a single “molecule” that will be reimaged when PBC’s are 
enforced. For example, by placing a distance restraint on a protein and the peptide it is 
complexed with, they become a single molecule in terms of PBCs. Confinement and Cartesian 
restraints work across periodic boundaries through OpenMM’s `periodicdistance()` function in 
the CustomExternalForce class.

.. |ang|    unicode:: U+00C5 .. ANGSTROM
