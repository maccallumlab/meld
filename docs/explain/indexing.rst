================
Indexing in MELD
================

When interacting with MELD we often need to give the index of a specific
*atom* or *residue*. This document describes how indexing works in MELD
and explains the various methods of indexing.

MELD uses zero-based indexing internally
----------------------------------------

MELD is based on the python programming language, which, like most modern programming languages,
uses zero-based indexing. However, in structural biology, we often use one-based indexing. The
difference is that zero-based indexing starts counting from zero, while one-based indexing starts
from one.

**Internally, MELD uses zero-based indexing**, but provides various methods for using
one-based indexing.

To help eliminate errors, all functions in meld that take an atom index require that it is
of type :code:`AtomIndex`. This is effectively just an integer, but it has be labeled as
an :code:`AtomIndex` to indicate that it is a zero-based absolute atom index. Similarly,
functions that take a residue index require that it has type :code:`ResidueIndex`.

Functions for indexing
----------------------

The two primary ways for indexing are both methods of the sytem object:

- :code:`system.index.atom(resid, atom_name, expected_resname=None, chainid=None, one_based=False)`
- :code:`system.index.residue(resid, expected_resname=None, chainid=None, one_based=False)`

Calls to :code:`inex.atom` will return a zero-based absolute :code:`AtomIndex`.
Calls to :code:`index.residue` will return a zero-based absolute :code:`ResidueIndex`.

Specifying :code:`resname` to catch errors
------------------------------------------

Indexing can be tricky and errors can result in strange behavior, as e.g. restraints
may be created between the wrong atoms.

To help catch errors, it is possible to specify :code:`expected_resname`. When
:code:`rexpcected_resname` is specified, calls to :code:`index.atom` and 
:code:`index.residue` will check that actual residue name that is found
matches :code:`expected_resname`.

Note that the residue names will be those after processing by ``tleap``, so they may not correspond
exactly to those in a pdb file. Normally, the :code:`expected_resname` will be three characters in all-caps,
e.g. :code:`"ALA"`.

Using one-based indexing
------------------------

By default both :code:`index.atom` and :code:`index.residue` use zero-based indexing,
where both :code:`chainid` and :code:`resid` start from zero. To use one-based indexing
set :code:`one_based=True`, which will cause both :code:`resid` and :code:`chainid` to
be interpreted as one-based.

Using relative indexing
-----------------------

By default, the :code:`resid` refers to *absolute* residue index, which starts from zero
(one for one-based indexing) and does not consider which chain the residue resides in.
The ordering of residues corresponds to the order that sub-systems were added when the system
was built.

If :code:`chainid` is set, then :code:`resid` refers to the relative index of a residue
within the corresponding chain. So, :code:`resid=0, chainid=0` would refer to the first residue
in the first chain (assuming zero-based indexing).

Ordering of :code:`chainids`
----------------------------

Chains are indexed sequentially starting from zero (one for one-based indexing). The order
of chains is partially determined by the order that sub-systems are added in.

When created by sequence, each sub-system corresponds to exactly one chain. When
created from a pdb file, each sub-system will have the same number of chains
as the pdb file has unique chain indentifiers. The ordering of the chains
is alphabetical with a blank chain identifier coming first, followed by "A", etc.

To be more concrete, consider the following example:

- A sub-system is added from sequence
- A second subsystem is added from a pdb file

  - The pdb file contains two chain identifiers, "A" and "B".

In this case, the :code:`chainid` would be defined as follows:

- **0**: the chain added by sequence
- **1**: chain "A" from the pdb file
- **2**: chain "B" from the pdb file

In some cases, MELD will add additional residues that were not present in either
the sequence or pdb file. Examples include extra residues added to encode RDC
alignment tensors, which are added the :code:`RdcAlignmentPatcher` and
solvent and ions that are added when explicit solvent calculations are specified.
These additional residues are considered to be in an additional chain that is
added in the final position.
