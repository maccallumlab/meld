=======================
How to use peak mapping
=======================

Overview
--------

MELD allows for the sampling of peak assignments, which
is refered to as mapping. This is typically used for
NMR experiments, where we have a series of peaks. Each
peak corresponds to a group of atoms.

We know the peaks and we know the groups of atoms they
may be assigned to, but we don't know the actual assignment.
We sample the possible assignments using Metropolis Monte Carlo,
which provides the assignments that are in best agreement with
the restraints (derived from data) and the force field.

For concreteness, we will consider the case of paramagnetic
relaxation enhancement data, where each peak is either
attenuated or not, depending on its proximity to a paramagnetic
spin label.

Creating mappers
----------------

First, we need to create one or more mappers:

.. code-block:: python

    mapper = system.create_map("hsqc", n_peaks=100, atom_names=["H", "N"])

You can add as many mappers as you like. For example, there might be one
for each chain if the NMR experiments were done in a way that resolves
the peaks for each chain separately.

Each mapper requires a name, ``hsqc`` in the example. The name must be
unique, but otherwise is not important.

The number of peaks must be defined. In this case 100, ranging from 0 to
99, inclusive.

Finally, the list of atom names associated with each peak must be specified.
Since this is an HSQC spectrum, we defined ``N`` and ``H``, but the atoms
listed here don't need to correspond only to the NMR experiment. For example,
below we only use the amide proton, so ``atom_names=["H"]`` would have
sufficed. Alternatively, we could have also included an additional atom,
e.g. ``atom_names=["N", "H", "CA"]``. 

Adding atom groups
------------------

Next, we must add all of the possible atom groups that could be assigned
to the peaks:

.. code-block:: python

    # residue 1
    mapper.add_atom_group(
        N=system.index.atom(0, "N"),
        H=system.index.atom(0, "H")
    )
    # residue 2
    mapper.add_atom_group(
        N=system.index.atom(1, "N"),
        H=system.index.atom(1, "H")
    )
    # ...

Normally, one would use a for loop to add the atom groups. Each call to
``add_atom_group`` must have arguments that match ``atom_names`` from
``create_map``. The values must be ``AtomIndex``, which are conveniently
returned by MELD's indexing functions.

Each group of atoms is potentially assigned to exactly one peak. Importantly,
all of the atoms in the group are assigned together, so in the example abovel
the ``N`` and ``H`` from residue 1 will always be assigned together, and so on.

Mismatch between the number of peaks and atom groups
----------------------------------------------------

The number of atom groups and peaks does not need to match.

As explained below, peaks can be used to define restraints. When the number of
atom groups is greater than the number of peaks, there will be some left over
atom groups that are unassigned. These unassigned atom groups will simply not
be involved in restraints, and thus will not contribute to the energy of the
system.

When the number of peaks is larger than the number of residues, the situation
is slighly more complex. In this case, there will be some peaks that do not
map to anything. Any restraints involving those peaks will be treated
specially by MELD, such that they do not contribute to the energy of the system.

Using peaks in restraints
-------------------------

Peak mapping is currently only supported for :class:`DistanceRestraint`. When
defining a distance restraint, peak mapping can be used to define either of the
atoms involved, rather than specifying a fixed atom index.

For example, consider that peak zero was attenuated when a spin label was placed
on residue 42. In this case we would have a distance restraint that favors short
distances between *whatever peak 0 mapps to* and residue 42:

.. code-block:: python

    r1 = system.restraints.create(
        "distance",
        scaler=scaler,
        ramp=ramp,
        atom1=mapper.get_mapping(peak_id=0, atom_name="H"),
        atom2=system.index.atom(42, "CA"),
        r1=0 * u.nanometer,
        r2=0 * u.nanometer,
        r3=1.5 * u.nanometer,
        r4=1.7 * u.nanometer,
        k=2500 * u.kilojoule_per_mole / u.nanometer **2
    )

On the other hand, consider that peak two was not attentuated. We would include
a restraint that favors large distances between *whatever peak two maps to* and
residue 42:

.. code-block:: python

    r2 = system.restraints.create(
        "distance",
        scaler=scaler,
        ramp=ramp,
        atom1=mapper.get_mapping(peak_id=1, atom_name="H"),
        atom2=system.index.atom(42, "CA"),
        r1=1.3 * u.nanometer,
        r2=1.5 * u.nanometer,
        r3=999 * u.nanometer,
        r4=999 * u.nanometer,
        k=2500 * u.kilojoule_per_mole / u.nanometer **2
    )

The restraints must, of course, be added to the system. For example,
as always active:

.. code-block:: python

    system.restraints.add_as_always_active_list([r1, r2])

Sampling assignments
--------------------

Finally, the sample assignments, one must set the ``mapping_mcmc_steps``
parameter of the :class:`RunOptions` object. This will perform the
specified number of MCMC steps after each round of molecular dynamics.