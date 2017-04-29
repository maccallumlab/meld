#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

'''Implements all of the restraints available in MELD.

This file implements restraints and related classes for MELD.
Restraints are the primary way that "extra" forces are added
into MELD simulations.

There are several important concepts: `restraints`, `groups`,
`collections`, `scalers`, `ramps`, `positioners`, and the
`restraint manager`.

Restraints
----------
Restraints represent "extra" forces that can be added into a
MELD simulation. There are many different types of restraints.
Each restraint object has a variety of parameters that describe
the strength of the force, the atoms involved, and so on.

There are two main types of restraints, ``SelectableRestraint``
and ``NonSelectableRestraint``, which have substantially different
behavior.

``NonSelectableRestraint`` are "always on". They may be scaled by
scalers and ramps, but the force from each ``NonSelectableRestraint``
is independent of other restraints.

``SelectableRestraint`` have forces and energies that depend on
other ``SelectableRestraint``. They may be combined into
``RestraintGroup``s, which allows for the ``n_active`` lowest
energy restraints to be active at each timestep. The remaining
restraints are inactive and do not contribute their forces or
energy to the system for that timestep. This selectable nature
allows for the implmentation of very flexible restraint
strategies useful for a variety of problems in structural
biology [1]_, [2]_.

The standard way to create a restraint is using their
``RestraintMangaer.create_restraint`` with the appropriate
restraint key:

>>> r = system.restraints.create_restraint(rest_key, params...)


Groups
------
``SelectableRestraint`` must be part of a ``RestraintGroup``. Each
timestep, the restraints are sorted by energy and the ``num_active``
restraints with the lowest energy are activated for the timestep,
while the rest are ignored. It is not possible to add a
``NonSelectableRestraint`` to a ``RestraintGroup``.

RestraintGroups are created by:

>>> g = system.restraints.create_restraint(list_of_restraints, num_active)

Collections
-----------
There are two types of collection: always on, and selectively active.

Restraints that will always be active are added to a single always on
collection. The standard ways to do this is:

>>> system.restraints.add_as_always_active(restraint)
>>> system.restraints.add_as_always_active_list(list_of_restraints)

Restraints or groups of restraints that will be selected are added
to selectively active collections. A mix of bare
``SelectableRestraint`` or ``RestraintGroup`` objects may be added.
When bare restraints are added, they are automatically placed into
a group containing only that with restraint with ``num_active=1``.
The standard way to create a restraint group is:

>>> system.restraints.add_selectively_active_collection(
        list_of_restraints_and_groups, num_active)

Scalers
-------
Each replica in a MELD simulation has a value ``alpha`` that
runs from 0.0 to 1.0, inclusive. The lowest replica always
has ``alpha=0``, while the highest has ``alpha=1``. The strength
of restraints can be scaled by specifying a Scaler that maps
alpha into a scaling of the force constant.

Scalers are created and added to a restraint by:

>>> scaler = system.restraints.create_scaler(scaler_key, params...)
>>> r = system.restraints.create_restraint(rest_key, scaler=scaler, params...)

Ramps
-----
Ramps are similar to Scalers, except that they map the step of
the simulation into a scaling of the force constant. They are
typically used to slowly turn on forces at the start of a simulation.

Ramps are created and added to a restraint by:

>>> ramp = system.restraints.create_scaler(ramp_key, params...)
>>> r = system.restraints.create_restraint(rest_key, ramp=ramp, params...)

**Note:** Ramps are created with the ``create_scaler`` method.

Positioners
-----------
Positioners are used to control the position or distance in a restraint. They
function similar to Scalers, but rather than returning a value in [0, 1], they
return a value from a defined range.

Positioners are created and added to a restraint by:

>>> positioner = system.restraints.create_scaler(pos_key, params...)
>>> r = system.restraints.create_restraint(
        rest_key, param=positioner, params...)

**Note:** Positioners are created with the ``create_scaler`` method.

Restraint Manager
-----------------
The ``System`` object maintains a ``RestraintManager`` object, which is the
primary means for interacting with restraints. Generally, restraints, groups,
scalers, etc are created through the ``RestraintManager``, rather than
by direct construction.

References
----------
.. [1] J.L. MacCallum, A. Perez, and K.A. Dill, Determining protein structures
       by combining semireliable data with atomistic physical models by Bayesian
       inference, PNAS, 2015, 112(22), pp.6985--6990.
.. [2] A. Perez, J.L. MacCallum, and K.A. Dill, Accelerating molecular simulations
       of proteins using Bayesian inference on weak information, PNAS, 2015,
       112(38), pp. 11846--11851.

'''
from __future__ import print_function
from six import with_metaclass
import math
import numpy as np
from collections import namedtuple


class _RestraintRegistry(type):
    """
    Metaclass that maintains a registry of restraint types.

    All classes that descend from Restraint inherit _RestraintRegistry as their
    metaclass. _RestraintRegistry will automatically maintain a map between
    the class attribute '_restraint_key_' and all restraint types.

    The function get_constructor_for_key is used to get the class for the
    corresponding key.
    """
    _restraint_registry = {}

    def __init__(cls, name, bases, attrs):
        if name in ['Restraint', 'SelectableRestraint',
                    'NonSelectableRestraint']:
            pass    # we don't register the base classes
        else:
            try:
                key = attrs['_restraint_key_']
            except KeyError:
                raise RuntimeError(
                    'Restraint type {} subclasses Restraint, '
                    'but does not set _restraint_key_'.format(name))
            if key in _RestraintRegistry._restraint_registry:
                raise RuntimeError(
                    'Trying to register two different classes'
                    'with _restraint_key_ = {}.'.format(key))
            _RestraintRegistry._restraint_registry[key] = cls

    @classmethod
    def get_constructor_for_key(self, key):
        """Get the constructor for the restraint type matching key."""
        try:
            return _RestraintRegistry._restraint_registry[key]
        except KeyError:
            raise RuntimeError(
                'Unknown restraint type "{}".'.format(key))


class Restraint(with_metaclass(_RestraintRegistry, object)):
    """Abstract class for all restraints."""
    pass


class SelectableRestraint(Restraint):
    """Abstract class for selectable restraints."""
    pass


class NonSelectableRestraint(Restraint):
    """Abstract class for non-selectable restraints."""
    pass


class DistanceRestraint(SelectableRestraint):
    """
    Restrain the distance between two groups

    The energy is zero between ``r2`` and ``r3``. It increases
    quadratically between ``r1`` and ``r2`` and between
    ``r3`` and ``r4``. The energy increases linearly below ``r1``
    and above ``r4``.

    Parameters
    ----------
    system : meld.system.System
             system object that restraint belongs to
    scaler : Scaler or None
             A Scaler to vary the force constant with alpha.
             If ``None``, then a constant 1.0 scaler will
             be used.
    atom_1_res_index : integer
                       residue index starting from 1
    atom_1_name : string
                  atom name
    atom_2_res_index : integer
                       residue index starting from 1
    atom_2_name : string
                  atom name
    r1 : float
         in nanometers
    r2 : float
         in nanometers
    r3 : float
         in nanometers
    r4 : float
         in nanometers
    k : float
        in :math:`kJ/mol/nm^2`

    Attributes
    ----------
    system : meld.system.System
             system object that restraint belongs to
    scaler : Scaler or None
             A Scaler to vary the force constant with alpha.
             If ``None``, then a constant 1.0 scaler will
             be used.
    ramp : Ramp or None
           A ramp to vary the force constant with simulation time.
           If ``None`` then a constant 1.0 ramp will be used.
    atom_index_1 : int
                   atom index starting from 1
    atom_index_2 : int
                   atom index starting from 1
    r1 : float
         in nanometers
    r2 : float
         in nanometers
    r3 : float
         in nanometers
    r4 : float
         in nanometers
    k : float
        in :math:`kJ/mol/nm^2`
    """

    _restraint_key_ = 'distance'

    def __init__(self, system, scaler, ramp, atom_1_res_index, atom_1_name,
                 atom_2_res_index, atom_2_name, r1, r2, r3, r4, k):
        self.atom_index_1 = system.index_of_atom(atom_1_res_index, atom_1_name)
        self.atom_index_2 = system.index_of_atom(atom_2_res_index, atom_2_name)
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.k = k
        self.scaler = scaler
        self.ramp = ramp
        self._check(system)

    def _check(self, system):
        if self.r1 < 0 or self.r2 < 0 or self.r3 < 0 or self.r4 < 0:
            raise RuntimeError(
                'r1 to r4 must be > 0. r1={} r2={} r3={} r4={}.'.format(
                    self.r1, self.r2, self.r3, self.r4))
        if self.r2 < self.r1:
            raise RuntimeError(
                'r2 must be >= r1. r1={} r2={}.'.format(self.r1, self.r2))
        if self.r3 < self.r2:
            raise RuntimeError(
                'r3 must be >= r2. r2={} r3={}.'.format(self.r2, self.r3))
        if self.r4 < self.r3:
            raise RuntimeError(
                'r4 must be >= r3. r3={} r4={}.'.format(self.r3, self.r4))
        if self.k < 0:
            raise RuntimeError('k must be >= 0. k={}.'.format(self.k))


class GMMDistanceRestraint(SelectableRestraint):
    """
    Restrain multiple distances using Gaussian mixture models

    The energy has the form:

    E = w1 N1 exp(-0.5 (r-u1)^T P1 (r-u1)) + w2 N2 exp(-0.5 (r-u2)^T P2 (r-u2)) + ...

    where:
       w1, w2, ... are the weights
       N1, N2, ... are automatically calculated normalization factors
       r is the vector of distances for the atom pairs
       u1, u2, ... are the mean vectors for each component
       P1, P2, ... are the precision (inverse covariance) matrices for each component

    Parameters
    ----------
    system : meld.system.System
             system object that restraint belongs to
    scaler : Scaler or None
             A Scaler to vary the force constant with alpha.
             If ``None``, then a constant 1.0 scaler will
             be used.
    n_distances : int
                  number of distances involved in GMM; max 32
    n_components : int
                   number of mixture components; max 32
    atoms : [(res_index, atom_name)]
            a list of (res_index, atom_name) tuples of length 2*n_distances
    weights : array-like, shape(n_components)
              The weights for the mixture components
    means : array-like, shape(n_components, n_distances)
            The means of each mixture component
    precisions : array-like, shape(n_components, n_distances, n_distances)
                 The precision (i.e. inverse covariance) of each mixture component

    Attributes
    ----------
    scaler : Scaler or None
             A Scaler to vary the force constant with alpha.
             If ``None``, then a constant 1.0 scaler will
             be used.
    ramp : Ramp or None
           A ramp to vary the force constant with simulation time.
           If ``None`` then a constant 1.0 ramp will be used.
    n_distances : int
                  number of distances involved in restraint
    n_components : int
                   number of mixture components
    atoms : [int]
                 a list of atom indices of length 2*n_distances
    weights : array-like, shape(n_components)
              The weights for the mixture components
    means : array-like, shape(n_components, n_distances)
            The means of each mixture component, in nm
    precisions : array-like, shape(n_components, n_distances, n_distances)
                 The precision (i.e. inverse covariance) of each mixture component, in nm^(-2)
    """

    _restraint_key_ = 'gmm'

    def __init__(self, system, scaler, ramp, n_distances, n_components,
                 atoms, weights, means, precisions):
        self.scaler = scaler
        self.ramp = ramp
        self.n_distances = n_distances
        self.n_components = n_components
        self.weights = weights
        self.means = means
        self.precisions = precisions
        self.atoms = None
        self._setup_atoms(atoms, system)
        self._check(system)

    @classmethod
    def from_params(cls, system, scaler, ramp, params):
        '''Create a GMMDistanceRestraint from a GMMParams object.'''
        return cls(system, scaler, ramp,
                   params.n_distances, params.n_components,
                   params.atoms, params.weights,
                   params.means, params.precisions)

    def _setup_atoms(self, pair_list, system):
        self.atoms = []
        for res_index, atom_name in pair_list:
            self.atoms.append(system.index_of_atom(res_index, atom_name))

    def _check(self, system):
        if len(self.atoms) != 2 * self.n_distances:
            raise RuntimeError(
                'len(atoms) must be 2*n_distances')
        if self.weights.shape[0] != self.n_components:
            raise RuntimeError(
                'weights must have shape (n_components,)')
        if self.means.shape != (self.n_components, self.n_distances):
            raise RuntimeError(
                'means must have shape (n_components, n_distances)')
        if self.precisions.shape != (self.n_components,
                                     self.n_distances,
                                     self.n_distances):
            raise RuntimeError(
                'precisions must have shape (n_components, n_distances, n_distances)')
        for i in range(self.n_components):
            if not np.allclose(self.precisions[i, :, :], self.precisions[i, :, :].T):
                raise RuntimeError(
                    'precision matrix must be symmetric')
        for i in range(self.n_components):
            # Perform a Cholesky decomposition on each precision matrix.
            # This will fail if the matrix is not positive definite.
            try:
                np.linalg.cholesky(self.precisions[i, :, :])
            except np.linalg.LinAlgError:
                raise RuntimeError(
                    'precision matrices must be positive definite')


class HyperbolicDistanceRestraint(SelectableRestraint):
    _restraint_key_ = 'hyperbolic'

    def __init__(self, system, scaler, ramp, atom_1_res_index, atom_1_name,
                 atom_2_res_index, atom_2_name, r1, r2, r3, r4, k, asymptote):
        self.atom_index_1 = system.index_of_atom(atom_1_res_index, atom_1_name)
        self.atom_index_2 = system.index_of_atom(atom_2_res_index, atom_2_name)
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.k = k
        self.asymptote = asymptote

        self._check(system)

    def _check(self, system):
        if self.r1 < 0 or self.r2 < 0 or self.r3 < 0 or self.r4 < 0:
            raise RuntimeError(
                'r1 to r4 must be > 0. r1={} r2={} r3={} r4={}.'.format(
                    self.r1, self.r2, self.r3, self.r4))

        if self.r2 < self.r1:
            raise RuntimeError(
                'r2 must be >= r1. r1={} r2={}.'.format(self.r1, self.r2))

        if self.r3 < self.r2:
            raise RuntimeError(
                'r3 must be >= r2. r2={} r3={}.'.format(self.r2, self.r3))

        if self.r4 < self.r3:
            raise RuntimeError(
                'r4 must be >= r3. r3={} r4={}.'.format(self.r3, self.r4))

        if self.r3 == self.r4:
            raise RuntimeError(
                'r4 cannot be equal to r3. r3={} r4={}.'.format(
                    self.r3, self.r4))

        if self.k < 0:
            raise RuntimeError('k must be >= 0. k={}.'.format(self.k))

        if self.asymptote < 0:
            raise RuntimeError(
                'asymptote must be >= 0. asymptote={}.'.format(self.asymptote))


class TorsionRestraint(SelectableRestraint):
    # """
    # A torsion restraint

    # :param system: System
    # :param scaler:  force scaler
    # :param atom_1_res_index: integer, starting from 1
    # :param atom_1_name: atom name
    # :param atom_2_res_index: integer, starting from 1
    # :param atom_2_name: atom name
    # :param atom_3_res_index: integer, starting from 1
    # :param atom_3_name: atom name
    # :param atom_4_res_index: integer, starting from 1
    # :param atom_4_name: atom name
    # :param phi: equilibrium value, degrees
    # :param delta_phi: flat within delta_phi, degrees
    # :param k: :math:`kJ/mol/degree^2`
    # """

    _restraint_key_ = 'torsion'

    def __init__(self, system, scaler, ramp, atom_1_res_index, atom_1_name,
                 atom_2_res_index, atom_2_name, atom_3_res_index, atom_3_name,
                 atom_4_res_index, atom_4_name,
                 phi, delta_phi, k):

        self.atom_index_1 = system.index_of_atom(atom_1_res_index, atom_1_name)
        self.atom_index_2 = system.index_of_atom(atom_2_res_index, atom_2_name)
        self.atom_index_3 = system.index_of_atom(atom_3_res_index, atom_3_name)
        self.atom_index_4 = system.index_of_atom(atom_4_res_index, atom_4_name)
        self.phi = phi
        self.delta_phi = delta_phi
        self.k = k
        self.scaler = scaler
        self.ramp = ramp
        self._check()

    def _check(self):
        if len(set([self.atom_index_1, self.atom_index_2, self.atom_index_3,
                    self.atom_index_4])) != 4:
            raise RuntimeError(
                'All four indices of a torsion restraint must be unique.')
        if self.phi < -180 or self.phi > 180:
            raise RuntimeError(
                '-180 <= phi <= 180. phi was {}.'.format(self.phi))
        if self.delta_phi < 0 or self.delta_phi > 180:
            raise RuntimeError(
                '0 <= delta_phi < 180. delta_phi was {}.'.format(
                    self.delta_phi))
        if self.k < 0:
            raise RuntimeError('k >= 0. k was {}.'.format(self.k))


class DistProfileRestraint(SelectableRestraint):
    _restraint_key_ = 'dist_prof'

    def __init__(self, system, scaler, ramp, atom_1_res_index, atom_1_name,
                 atom_2_res_index, atom_2_name, r_min, r_max, n_bins,
                 spline_params, scale_factor):
        self.scaler = scaler
        self.ramp = ramp
        self.atom_index_1 = system.index_of_atom(atom_1_res_index, atom_1_name)
        self.atom_index_2 = system.index_of_atom(atom_2_res_index, atom_2_name)
        self.r_min = r_min
        self.r_max = r_max
        self.n_bins = n_bins
        self.spline_params = spline_params
        self.scale_factor = scale_factor
        self._check()

    def _check(self):
        assert self.r_min >= 0.
        assert self.r_max > self.r_min
        assert self.n_bins > 0
        assert self.spline_params.shape[0] == self.n_bins
        assert self.spline_params.shape[1] == 4


class TorsProfileRestraint(SelectableRestraint):
    _restraint_key_ = 'tors_prof'

    def __init__(self, system, scaler, ramp,
                 atom_1_res_index, atom_1_name, atom_2_res_index, atom_2_name,
                 atom_3_res_index, atom_3_name, atom_4_res_index, atom_4_name,
                 atom_5_res_index, atom_5_name, atom_6_res_index, atom_6_name,
                 atom_7_res_index, atom_7_name, atom_8_res_index, atom_8_name,
                 n_bins, spline_params, scale_factor):
        self.scaler = scaler
        self.ramp = ramp
        self.atom_index_1 = system.index_of_atom(atom_1_res_index, atom_1_name)
        self.atom_index_2 = system.index_of_atom(atom_2_res_index, atom_2_name)
        self.atom_index_3 = system.index_of_atom(atom_3_res_index, atom_3_name)
        self.atom_index_4 = system.index_of_atom(atom_4_res_index, atom_4_name)
        self.atom_index_5 = system.index_of_atom(atom_5_res_index, atom_5_name)
        self.atom_index_6 = system.index_of_atom(atom_6_res_index, atom_6_name)
        self.atom_index_7 = system.index_of_atom(atom_7_res_index, atom_7_name)
        self.atom_index_8 = system.index_of_atom(atom_8_res_index, atom_8_name)
        self.n_bins = n_bins
        self.spline_params = spline_params
        self.scale_factor = scale_factor
        self._check()

    def _check(self):
        assert self.n_bins > 0
        n_params = self.n_bins * self.n_bins
        assert self.spline_params.shape[0] == n_params
        assert self.spline_params.shape[1] == 16


class RdcRestraint(NonSelectableRestraint):
    # """
    # Residual Dipolar Coupling Restraint

    # :param system: a System object
    # :param scaler: a force scaler
    # :param atom_1_res_index: integer, starting from 1
    # :param atom_1_name: atom name
    # :param atom_2_res_index:  integer, starting from 1
    # :param atom_2_name: atom name
    # :param kappa: prefactor for RDC calculation in :math:`Hz / Angstrom^3`
    # :param d_obs: observed dipolar coupling in Hz
    # :param tolerance: calculed couplings within tolerance (in Hz) of d_obs
    #                   will have zero energy and force
    # :param force_const: force sonstant in :math:`kJ/mol/Hz^2`
    # :param weight: dimensionless weight to place on this restraint
    # :param expt_index: integer experiment id

    # Typical values for kappa are:

    # - 1H - 1H: :math:`-360300 \ Hz / Angstrom^3`
    # - 13C - 1H: :math:`-90600 \ Hz / Angstrom^3`
    # - 15N - 1H: :math:`36500 \ Hz / Angstrom^3`

    # """

    _restraint_key_ = 'rdc'

    def __init__(self, system, scaler, ramp, atom_1_res_index, atom_1_name,
                 atom_2_res_index, atom_2_name, kappa, d_obs, tolerance,
                 force_const, weight, expt_index):
        self.atom_index_1 = system.index_of_atom(atom_1_res_index, atom_1_name)
        self.atom_index_2 = system.index_of_atom(atom_2_res_index, atom_2_name)
        self.kappa = float(kappa)
        self.d_obs = float(d_obs)
        self.tolerance = float(tolerance)
        self.force_const = float(force_const)
        self.weight = float(weight)
        self.expt_index = int(expt_index)
        self.scaler = scaler
        self.ramp = ramp
        self._check(system)

    def _check(self, system):
        if self.atom_index_1 == self.atom_index_2:
            raise ValueError('atom1 and atom2 must be different')
        if self.tolerance < 0:
            raise ValueError('tolerance must be > 0')
        if self.force_const < 0:
            raise ValueError('force_constant must be > 0')
        if self.weight < 0:
            raise ValueError('weight must be > 0')


class ConfinementRestraint(NonSelectableRestraint):
    # """
    # Confinement restraint

    # :param system: a System object
    # :param scaler: a force scaler
    # :param res_index: integer, starting from 1
    # :param atom_name: atom name
    # :param raidus: calculed couplings within tolerance (in Hz) of d_obs will
    #                have zero energy and force
    # :param force_const: force sonstant in :math:`kJ/mol/Hz^2`

    # Confines an atom to be within radius of the origin. These restraints are
    # typically set to somewhat larger than the expected radius of gyration of
    # the protein and help to keep the structures comapact even when the protein
    # is unfolded. Typically used with a ConstantScaler.

    # """

    _restraint_key_ = 'confine'

    def __init__(self, system, scaler, ramp, res_index, atom_name,
                 radius, force_const):
        self.atom_index = system.index_of_atom(res_index, atom_name)
        self.radius = float(radius)
        self.force_const = float(force_const)
        self.scaler = scaler
        self.ramp = ramp
        self._check(system)

    def _check(self, system):
        if self.radius < 0:
            raise ValueError('radius must be > 0')
        if self.force_const < 0:
            raise ValueError('force_constant must be > 0')


class CartesianRestraint(NonSelectableRestraint):
    '''Cartesian restraint on xyz coordinates'''

    _restraint_key_ = 'cartesian'

    def __init__(self, system, scaler, ramp, res_index, atom_name, x, y, z,
                 delta, force_const):
        self.atom_index = system.index_of_atom(res_index, atom_name)
        self.x = x
        self.y = y
        self.z = z
        self.delta = delta
        self.force_const = force_const
        self.scaler = scaler
        self.ramp = ramp
        self._check()

    def _check(self):
        if self.delta < 0:
            raise ValueError('delta must be non-negative')
        if self.force_const < 0:
            raise ValueError('force_const must be non-negative')


class YZCartesianRestraint(NonSelectableRestraint):
    '''Cartesian restraint on yz coordinates only'''

    _restraint_key_ = 'yzcartesian'

    def __init__(self, system, scaler, ramp, res_index, atom_name, y, z,
                 delta, force_const):
        self.atom_index = system.index_of_atom(res_index, atom_name)
        self.y = y
        self.z = z
        self.delta = delta
        self.force_const = force_const
        self.scaler = scaler
        self.ramp = ramp
        self._check()

    def _check(self):
        if self.delta < 0:
            raise ValueError('delta must be non-negative')
        if self.force_const < 0:
            raise ValueError('force_const must be non-negative')


class AbsoluteCOMRestraint(NonSelectableRestraint):
    ''' Restraint on the distance between a group and a point in space

    This class implements a restraint on the distance between the
    center of a group and a point in space.

    The weights used to calculate the center can be specified as
    ``weights``. If ``None``, then the masses of the atoms will be used.

    The ``dims`` parameter controls which dimensions are used to compute the
    distance. For example if ``dims='xyz'``, then the distance will be the
    normal distance in all three dimensions. If ``dims=x``, then only the
    x-component will be considered.

    Restraints are typically added using ``RestraintMangager.create_restraint``
    with the ``'abs_com'`` key:

    >>> r = system.restraints.create_restraint('abs_com',
                                               scaler=scaler, ramp=ramp,
                                               group=group,
                                               weights=weights,
                                               dims=dims,
                                               force_const=force_const,
                                               position=position)

    Parameters
    ----------
    system : meld.system.System
             system object used for indexing
    scaler : Scaler or None
             scaler for force constant
    ramp : Ramp or None
           ramp for force constant
    group : list of tuple
            [(res_index, atom_name), (res_index, atom_name)...]
    weights : array_like
              Weights to use when calculating the COM. If ``None``, then
              the masses will be used.
    dims : string
           combination of x, y, z that determines which dimensions
           are used when calculating the distance
    force_const : float
                  force constant in kJ/mol/nm^2
    point : array_like
            location in space to restrain to

    Attributes
    ----------
    scaler : Scaler
             scaler for the force constant
    ramp : Ramp
           ramp for the force
    indices : list
              index of atoms in group
    weights : array_like or None
              Weights used in COM calculation. If ``None``, then the
              masses will be used.
    dims : string
           combination of xyz dimensions used for distance calculation
    force_const : float
                  force constant
    position : array_like
               point in space that group is restrained to

    '''
    _restraint_key_ = 'abs_com'

    def __init__(self, system, scaler, ramp, group, weights, dims,
                 force_const, position):
        self.scaler = scaler
        self.ramp = ramp

        self.dims = dims
        self._check_dims()

        self.force_const = force_const
        if self.force_const < 0:
            raise ValueError('force_const cannot be negative')

        self.position = np.array(position, dtype=float)
        if len(self.position) != 3:
            raise ValueError(
                'position should be an array of [x, y, z]')

        self.weights = weights
        self.indices = self._get_indices(system, group)
        self._check_weights()

    def _check_weights(self):
        if self.weights is not None:
            if len(self.indices) != len(self.weights):
                raise ValueError(
                    'weights and group have different lengths')
            for w in self.weights:
                if w < 0:
                    raise ValueError(
                        'weights must be > 0')

    def _check_dims(self):
        for c in self.dims:
            if c not in 'xyz':
                raise ValueError(
                    'dims must be a combination of "xyz", found {}'.format(c))
        for c in 'xyz':
            if self.dims.count(c) > 1:
                raise ValueError(
                    '{} occurs more than once in dims'.format(c))

    def _get_indices(self, system, group):
        return [system.index_of_atom(res_ind, atom_name)
                for (res_ind, atom_name) in group]


class COMRestraint(NonSelectableRestraint):
    ''' Restraint on the distance between two groups along selected axes

    This class implements a restraint on the distance between the center of
    two groups.

    The weights used to calculate the center can be specified as ``weights1``
    and ``weights2``. If these are ``None``, then the masses of the atoms
    will be used.

    The ``dims`` parameter controls which dimensions are used to compute the
    distance. For example if ``dims='xyz'``, then the distance will be the
    normal distance in all three dimensions. If ``dims='x'``, then only the
    x-component will be considered.

    Restraints are typically added using ``RestraintMangager.create_restraint``
    with the ``'com'`` key:

    >>> r = system.restraints.create_restraint('com', scaler, ramp=ramp,
                                               group1=group1, group2=group2,
                                               weights1=weights1,
                                               weights2=weights2,
                                               dims=dims,
                                               force_const=force_const,
                                               distance=distance)

    Parameters
    ----------
    system : meld.system.System
             system object used for indexing
    scaler : Scaler or None
             scaler for force constant
    ramp : Ramp or None
           ramp for force constant
    group1 : list of tuple
             [(res_index, atom_name), (res_index, atom_name)...]
    group2 : list of tuple
             [(res_index, atom_name), (res_index, atom_name)...]
    weights1 : array_like
               Weights to use when calculating the COM. If ``None``,
               then the atom masses will be used.
    weights2 : array_like
               Weights to use when calculating the COM. If ``None``,
               then the atom masses will be used.
    dims : string
           combination of x, y, z that determines which dimensions
           are used when calculating the distance
    force_const : float
                  force constant in kJ/mol/nm^2
    distance : float or Positioner
               distance between groups

    Attributes
    ----------
    scaler : Scaler
             scaler for the force constant
    ramp : Ramp
           ramp for the force
    indices1 : list
               index of atoms in group 1
    indices2 : list
               index of atoms in group 2
    weights1 : array_like
               Weights to use when calculating the COM. If ``None``,
               then the atom masses will be used.
    weights2 : array_like
               Weights to use when calculating the COM. If ``None``,
               then the atom masses will be used.
    dims : string
           combination of xyz dimensions used for distance calculation
    force_const : float
                  force constant
    positioner : Positioner
                 controls the distance between groups

    '''
    _restraint_key_ = 'com'

    def __init__(self, system, scaler, ramp, group1, group2, weights1,
                 weights2, dims, force_const, distance):
        # setup indices
        self.scaler = scaler
        self.ramp = ramp

        self.indices1 = self._get_indices(system, group1)
        self.indices2 = self._get_indices(system, group2)

        # setup the weights
        self.weights1 = weights1
        if self.weights1 is not None:
            if len(self.indices1) != len(self.weights1):
                raise ValueError(
                    'len(indices1) != len(weights1)')
            for w in self.weights1:
                if w < 0:
                    raise ValueError(
                        'weights1 must be > 0')
        self.weights2 = weights2
        if weights2 is not None:
            if len(self.indices2) != len(weights2):
                raise ValueError(
                    'len(indices2) != len(weights2)')
            for w in self.weights2:
                if w < 0:
                    raise ValueError(
                        'weights2 must be > 0')

        # setup the dimensions
        self.dims = dims
        self._check_dims()

        # setup the force constant and positioner
        self.force_const = force_const
        if self.force_const < 0:
            raise ValueError(
                'force constant cannot be negative')
        if isinstance(distance, Positioner):
            self.positioner = distance
        else:
            if distance < 0.:
                raise ValueError(
                    'distance cannot be negative')

            self.positioner = ConstantPositioner(distance)

    def _get_indices(self, system, group):
        return [system.index_of_atom(res_ind, atom_name)
                for (res_ind, atom_name) in group]

    def _check_dims(self):
        # check for non 'xyz'
        for c in self.dims:
            if c not in 'xyz':
                raise ValueError(
                    'dims must be a combination of "xyz", found {}'.format(c))
        for dim in 'xyz':
            count = self.dims.count(dim)
            if count > 1:
                raise ValueError(
                    '{} occurs more than once in dims'.format(dim))


class AlwaysActiveCollection(object):
    '''
    '''
    def __init__(self):
        self._restraints = []

    @property
    def restraints(self):
        return self._restraints

    def add_restraint(self, restraint):
        if not isinstance(restraint, Restraint):
            raise RuntimeError(
                'Tried to add unknown restraint of type {}.'.format(
                    str(type(restraint))))
        self._restraints.append(restraint)


class SelectivelyActiveCollection(object):
    '''
    '''
    def __init__(self, restraint_list, num_active):
        self._groups = []
        if not restraint_list:
            raise RuntimeError(
                'SelectivelyActiveCollection cannot have empty'
                'restraint list.')
        for rest in restraint_list:
            self._add_restraint(rest)

        if num_active < 0:
            raise RuntimeError('num_active must be >= 0.')
        n_rest = len(self._groups)
        if num_active > n_rest:
            raise RuntimeError(
                'num active must be <= num_groups ({}).'.format(n_rest))
        self._num_active = num_active

    @property
    def groups(self):
        return self._groups

    @property
    def num_active(self):
        return self._num_active

    def _add_restraint(self, restraint):
        if isinstance(restraint, RestraintGroup):
            self._groups.append(restraint)
        elif not isinstance(restraint, SelectableRestraint):
            raise RuntimeError(
                'Cannot add restraint of type {} to'
                'SelectivelyActiveCollection'.format(str(type(restraint))))
        else:
            group = RestraintGroup([restraint], 1)
            self._groups.append(group)


class RestraintGroup(object):
    def __init__(self, rest_list, num_active):
        self._restraints = []
        if not rest_list:
            raise RuntimeError('rest_list cannot be empty.')
        for rest in rest_list:
            self._add_restraint(rest)

        if num_active < 0:
            raise RuntimeError('num_active must be >= 0.')
        n_rest = len(self._restraints)
        if num_active > n_rest:
            raise RuntimeError(
                'num_active must be <= n_rest ({}).'.format(n_rest))
        self._num_active = num_active

    @property
    def restraints(self):
        return self._restraints

    @property
    def num_active(self):
        return self._num_active

    def _add_restraint(self, rest):
        if not isinstance(rest, SelectableRestraint):
            raise RuntimeError(
                'Can only add SelectableRestraints to a RestraintGroup.')
        self._restraints.append(rest)


class RestraintManager(object):
    '''
    '''
    def __init__(self, system):
        self._system = system
        self._always_active = AlwaysActiveCollection()
        self._selective_collections = []

    @property
    def always_active(self):
        return self._always_active.restraints

    @property
    def selectively_active_collections(self):
        return self._selective_collections

    def add_as_always_active(self, restraint):
        self._always_active.add_restraint(restraint)

    def add_as_always_active_list(self, restraint_list):
        for r in restraint_list:
            self.add_as_always_active(r)

    def add_selectively_active_collection(self, rest_list, num_active):
        self._selective_collections.append(
            SelectivelyActiveCollection(rest_list, num_active))

    def create_restraint(self, rest_type, scaler=None, ramp=None, **kwargs):
        if scaler is None:
            scaler = ConstantScaler()
        else:
            if not isinstance(scaler, RestraintScaler):
                raise ValueError(
                    'scaler must be a subclass of RestraintScaler, '
                    'you tried to add a {}.'.format(type(scaler)))

        if ramp is None:
            ramp = ConstantRamp()
        else:
            if not isinstance(ramp, TimeRamp):
                raise ValueError(
                    'ramp must be a subclass of TimeRamp,'
                    'you tried to add a {}.'.format(type(ramp)))

        return _RestraintRegistry.get_constructor_for_key(rest_type)(
            self._system, scaler, ramp, **kwargs)

    def create_restraint_group(self, rest_list, num_active):
        return RestraintGroup(rest_list, num_active)

    def create_scaler(self, scaler_type, **kwargs):
        return ScalerRegistry.get_constructor_for_key(scaler_type)(**kwargs)


class ScalerRegistry(type):
    # '''
    # Metaclass that maintains a registry of scaler types.

    # All classes that decend from Scaler inherit ScalerRegistry as their
    # metaclass. ScalerRegistry will automatically maintain a map between
    # the class attribute '_scaler_key_' and all scaler types.

    # The function get_constructor_for_key is used to get the class for the
    # corresponding key.
    # '''
    _scaler_registry = {}

    def __init__(cls, name, bases, attrs):
        if name in ['AlphaMapper', 'RestraintScaler',
                    'TimeRamp', 'Positioner']:
            pass    # we don't register the base classes
        else:
            try:
                key = attrs['_scaler_key_']
            except KeyError:
                raise RuntimeError(
                    'Scaler type {} subclasses Scaler, but'
                    'does not set _scaler_key_'.format(name))
            if key in ScalerRegistry._scaler_registry:
                raise RuntimeError(
                    'Trying to register two different classes'
                    'with _scaler_key_ = {}.'.format(key))
            ScalerRegistry._scaler_registry[key] = cls

    @classmethod
    def get_constructor_for_key(self, key):
        '''Get the constructor for the scaler type matching key.'''
        try:
            return ScalerRegistry._scaler_registry[key]
        except KeyError:
            raise RuntimeError(
                'Unknown scaler type "{}".'.format(key))


class AlphaMapper(with_metaclass(ScalerRegistry, object)):
    '''Base class for all scalers.'''
    pass

    def _check_alpha_range(self, alpha):
        if alpha < 0 or alpha > 1:
            raise RuntimeError('0 >= alpha >= 1. alpha is {}.'.format(alpha))

    def _handle_boundaries(self, alpha):
        if alpha <= self._alpha_min:
            return 1.
        elif alpha >= self._alpha_max:
            return 0.
        else:
            return None

    def _check_alpha_min_max(self):
        if (self._alpha_min < 0 or self._alpha_min > 1 or
                self._alpha_max < 0 or self._alpha_max > 1):
            raise RuntimeError(
                'alpha_min and alpha_max must be in range [0, 1].'
                'alpha_min={} alpha_max={}.'.format(
                    self._alpha_min, self._alpha_max))
        if self._alpha_min >= self._alpha_max:
            raise RuntimeError(
                'alpha_max must be less than alpha_min.'
                'alpha_min={} alpha_max={}.'.format(
                    self._alpha_min, self._alpha_max))


class RestraintScaler(AlphaMapper):
    '''Base class for all resraint scaler classes.'''


class ConstantScaler(RestraintScaler):
    '''This scaler is "always on" and always returns a value of 1.0".'''

    _scaler_key_ = 'constant'

    def __call__(self, alpha):
        self._check_alpha_range(alpha)
        return 1.0


class LinearScaler(RestraintScaler):
    '''This scaler linearly interpolates between 0 and 1 from alpha_min
    to alpha_max.'''

    _scaler_key_ = 'linear'

    def __init__(self, alpha_min, alpha_max, strength_at_alpha_min=1.0,
                 strength_at_alpha_max=0.0):
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._strength_at_alpha_min = strength_at_alpha_min
        self._strength_at_alpha_max = strength_at_alpha_max
        self._delta = alpha_max - alpha_min
        self._check_alpha_min_max()

    def __call__(self, alpha):
        self._check_alpha_range(alpha)
        scale = self._handle_boundaries(alpha)
        if scale is None:
            scale = 1.0 - (alpha - self._alpha_min) / self._delta
        scale = (
            (1.0 - scale) *
            (self._strength_at_alpha_max - self._strength_at_alpha_min) +
            self._strength_at_alpha_min)
        return scale


class PlateauLinearScaler(RestraintScaler):
    # r'''This scaler linearly interpolates between 0 and 1 from alpha_min to
    # alpha_one, keeps the value of 1 until alpha_two and then decreases
    # linearly until 0 in alpha_max.

    #     ------   strength alpha_min --> between two and one
    #   /        \
    #  /          \ strength alpha_max --> > alpha_max and
    #                                        below alphamin
    # '''

    _scaler_key_ = 'plateau'

    def __init__(self, alpha_min, alpha_one, alpha_two, alpha_max,
                 strength_at_alpha_min=1.0, strength_at_alpha_max=0.0):
        self._alpha_min = float(alpha_min)
        self._alpha_one = float(alpha_one)
        self._alpha_two = float(alpha_two)
        self._alpha_max = float(alpha_max)
        self._strength_at_alpha_min = strength_at_alpha_min
        self._strength_at_alpha_max = strength_at_alpha_max
        self._check_alpha_min_max()

    def __call__(self, alpha):
        self._check_alpha_range(alpha)
        if alpha <= self._alpha_min:
            scale = self._strength_at_alpha_max
        else:
            if alpha <= self._alpha_one:
                # Decreasing
                scale = (
                    1.0 - (self._alpha_one - alpha) /
                    (self._alpha_one - self._alpha_min))
                scale = (
                    scale *
                    (self._strength_at_alpha_min -
                     self._strength_at_alpha_max) +
                    self._strength_at_alpha_max)

            elif alpha <= self._alpha_two:
                scale = self._strength_at_alpha_min
            elif alpha <= self._alpha_max:
                # Increasing
                scale = (1.0 - (alpha - self._alpha_two) /
                         (self._alpha_max - self._alpha_two))
                scale = ((1.0 - scale) *
                         (self._strength_at_alpha_max -
                          self._strength_at_alpha_min) +
                         self._strength_at_alpha_min)
            else:
                scale = self._strength_at_alpha_max
        return scale


class NonLinearScaler(RestraintScaler):
    _scaler_key_ = 'nonlinear'

    def __init__(self, alpha_min, alpha_max, factor, strength_at_alpha_min=1.0,
                 strength_at_alpha_max=0.0):
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._strength_at_alpha_min = strength_at_alpha_min
        self._strength_at_alpha_max = strength_at_alpha_max
        self._check_alpha_min_max()
        if factor < 1:
            raise RuntimeError(
                'factor must be >= 1. factor={}.'.format(factor))
        self._factor = factor

    def __call__(self, alpha):
        self._check_alpha_range(alpha)
        scale = self._handle_boundaries(alpha)
        if scale is None:
            delta = ((alpha - self._alpha_min) /
                     (self._alpha_max - self._alpha_min))
            norm = 1.0 / (math.exp(self._factor) - 1.0)
            scale = norm * (math.exp(self._factor * (1.0 - delta)) - 1.0)
        scale = ((1.0 - scale) *
                 (self._strength_at_alpha_max - self._strength_at_alpha_min) +
                 self._strength_at_alpha_min)
        return scale


class PlateauNonLinearScaler(RestraintScaler):
    # '''This scaler linearly interpolates between 0 and 1 from alpha_min
    # to alpha_one, keeps the value of 1 until alpha_two and then decreases
    # linearly until 0 in alpha_max.'''

    _scaler_key_ = 'plateaunonlinear'

    def __init__(self, alpha_min, alpha_one, alpha_two, alpha_max, factor,
                 strength_at_alpha_min=1.0, strength_at_alpha_max=0.0):
        self._alpha_min = float(alpha_min)
        self._alpha_one = float(alpha_one)
        self._alpha_two = float(alpha_two)
        self._alpha_max = float(alpha_max)
        self._strength_at_alpha_min = strength_at_alpha_min
        self._strength_at_alpha_max = strength_at_alpha_max
        self._check_alpha_min_max()
        if factor < 1:
            raise RuntimeError(
                'factor must be >= 1. factor={}.'.format(factor))
        self._factor = factor

    def __call__(self, alpha):
        self._check_alpha_range(alpha)
        if alpha <= self._alpha_min:
            scale = self._strength_at_alpha_max
        else:
            if alpha <= self._alpha_one:
                delta = ((alpha - self._alpha_min) /
                         (self._alpha_one - self._alpha_min))
                norm = 1.0 / (math.exp(self._factor) - 1.0)
                scale = norm * (math.exp(self._factor * (1.0 - delta)) - 1.0)
                scale = ((1.0 - scale) *
                         (self._strength_at_alpha_min -
                          self._strength_at_alpha_max) +
                         self._strength_at_alpha_max)
            elif alpha <= self._alpha_two:
                scale = self._strength_at_alpha_min
            elif alpha <= self._alpha_max:
                delta = ((alpha - self._alpha_two) /
                         (self._alpha_max - self._alpha_two))
                norm = 1.0 / (math.exp(self._factor) - 1.0)
                scale = norm * (math.exp(self._factor * (1.0 - delta)) - 1.0)
                scale = ((1.0 - scale) *
                         (self._strength_at_alpha_max -
                          self._strength_at_alpha_min) +
                         self._strength_at_alpha_min)
            else:
                scale = self._strength_at_alpha_max

        return scale


class PlateauSmoothScaler(RestraintScaler):
    # '''This scaler linearly interpolates between 0 and 1 from alpha_min
    # to alpha_one, keeps the value of 1 until alpha_two and then decreases
    # linearly until 0 in alpha_max.'''
    _scaler_key_ = 'plateausmooth'

    def __init__(self, alpha_min, alpha_one, alpha_two, alpha_max,
                 strength_at_alpha_min=1.0, strength_at_alpha_max=0.0):
        self._alpha_min = float(alpha_min)
        self._alpha_one = float(alpha_one)
        self._alpha_two = float(alpha_two)
        self._alpha_max = float(alpha_max)
        self._strength_at_alpha_min = strength_at_alpha_min
        self._strength_at_alpha_max = strength_at_alpha_max
        self._check_alpha_min_max()

    def __call__(self, alpha):
        self._check_alpha_range(alpha)
        if alpha <= self._alpha_min:
            scale = self._strength_at_alpha_max
        else:
            if alpha <= self._alpha_one:
                delta = ((alpha - self._alpha_min) /
                         (self._alpha_one - self._alpha_min))
                scale = delta*delta*(3-2*delta)
                scale = ((1.0 - scale) *
                         (self._strength_at_alpha_max -
                          self._strength_at_alpha_min) +
                         self._strength_at_alpha_min)
            elif alpha <= self._alpha_two:
                scale = self._strength_at_alpha_min
            elif alpha <= self._alpha_max:
                delta = ((alpha - self._alpha_two) /
                         (self._alpha_max - self._alpha_two))
                scale = 1 + delta*delta*(2*delta-3)
                scale = ((1.0 - scale) *
                         (self._strength_at_alpha_max -
                          self._strength_at_alpha_min) +
                         self._strength_at_alpha_min)
            else:
                scale = self._strength_at_alpha_max
        return scale


class GeometricScaler(RestraintScaler):
    _scaler_key_ = 'geometric'

    def __init__(self, alpha_min, alpha_max, strength_at_alpha_min,
                 strength_at_alpha_max):
        self._alpha_min = float(alpha_min)
        self._alpha_max = float(alpha_max)
        self._strength_at_alpha_min = float(strength_at_alpha_min)
        self._strength_at_alpha_max = float(strength_at_alpha_max)
        self._delta_alpha = self._alpha_max - self._alpha_min
        self._check_alpha_min_max()

    def __call__(self, alpha):
        self._check_alpha_range(alpha)

        if alpha < 0 or alpha > 1:
            raise RuntimeError('0 <= alpha <=1 1')

        elif alpha <= self._alpha_min:
            return self._strength_at_alpha_min

        elif alpha <= self._alpha_max:
            frac = (alpha - self._alpha_min) / self._delta_alpha
            delta = (math.log(self._strength_at_alpha_max) -
                     math.log(self._strength_at_alpha_min))
            return math.exp(delta * frac +
                            math.log(self._strength_at_alpha_min))

        else:
            return self._strength_at_alpha_max


class TimeRamp(AlphaMapper):
    '''Base class for all time ramp classes.'''


class ConstantRamp(TimeRamp):
    '''TimeRamp that always returns 1.0'''
    _scaler_key_ = 'constant_ramp'

    def __call__(self, timestep):
        if timestep < 0:
            raise ValueError('Timestep is < 0.')
        return 1.0


class LinearRamp(TimeRamp):
    '''TimeRamp that interpolates linearly'''

    _scaler_key_ = 'linear_ramp'

    def __init__(self, start_time, end_time, start_weight, end_weight):
        self.t_start = float(start_time)
        self.t_end = float(end_time)
        self.w_start = float(start_weight)
        self.w_end = float(end_weight)

    def __call__(self, timestep):
        if timestep < 0:
            raise ValueError('Timestep is < 0.')
        if timestep < self.t_start:
            return self.w_start
        elif timestep < self.t_end:
            return (self.w_start + (self.w_end - self.w_start) *
                    (float(timestep) - self.t_start) /
                    (self.t_end - self.t_start))
        else:
            return self.w_end


class NonLinearRamp(TimeRamp):
    '''TimeRamp that interpolates non-linearly'''

    _scaler_key_ = 'nonlinear_ramp'

    def __init__(self, start_time, end_time, start_weight, end_weight, factor):
        if end_time <= start_time:
            raise ValueError('end_time must be > start_time')
        if factor < 1.0:
            raise ValueError('factor myst be > 1.0')

        self.t_start = float(start_time)
        self.t_end = float(end_time)
        self.w_start = float(start_weight)
        self.w_end = float(end_weight)
        self.factor = float(factor)

    def __call__(self, timestep):
        if timestep < 0:
            raise ValueError('timestep is < 0.')

        if timestep < self.t_start:
            return self.w_start
        elif timestep < self.t_end:
            # we scale differently depending on if we are ramping up or down
            # we change more slowly at lower values and more rapidly at
            # higher values
            #
            # this is for scaling up
            if self.w_end > self.w_start:
                delta = (1.0 -
                         (float(timestep) - self.t_start) /
                         (self.t_end - self.t_start))
                norm = 1.0 / (math.exp(self.factor) - 1.0)
                scale = norm * (math.exp(self.factor * (1.0 - delta)) - 1.0)
                return scale * (self.w_end - self.w_start) + self.w_start
            # this is for scaling down
            else:
                delta = ((float(timestep) -
                          self.t_start) /
                         (self.t_end - self.t_start))
                norm = 1.0 / (math.exp(self.factor) - 1.0)
                scale = norm * (math.exp(self.factor * (1.0 - delta)) - 1.0)
                return ((1.0 - scale) *
                        (self.w_end - self.w_start) +
                        self.w_start)
        else:
            return self.w_end


class TimeRampSwitcher(TimeRamp):
    # '''
    # Switches between two TimeRamp objects.

    # Class first_ramp before switching time. At the switching
    # time it switches to second_ramp, which it uses thereafter.
    # '''

    _scaler_key_ = 'ramp_switcher'

    def __init__(self, first_ramp, second_ramp, switching_time):
        self.first_ramp = first_ramp
        self.second_ramp = second_ramp
        self.switching_time = switching_time

    def __call__(self, timestep):
        if timestep < self.switching_time:
            return self.first_ramp(timestep)
        else:
            return self.second_ramp(timestep)


class Positioner(AlphaMapper):
    '''Base class for all positioner classes.'''


class ConstantPositioner(Positioner):
    '''Always returns the supplied value.'''

    _scaler_key_ = 'constant_positioner'

    def __init__(self, value):
        self._value = value

    def __call__(self, alpha):
        if alpha < 0:
            raise ValueError('alpha must be >= 0')
        if alpha > 1:
            raise ValueError('alpha must be <= 1')

        return self._value


class LinearPositioner(Positioner):
    '''Position restraints linearly within a range'''

    _scaler_key_ = 'linear_positioner'

    def __init__(self, alpha_min, alpha_max, pos_min, pos_max):
        if alpha_max <= alpha_min:
            raise ValueError('alpha_max must be > alpha_min')

        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.pos_min = float(pos_min)
        self.pos_max = float(pos_max)

    def __call__(self, alpha):
        if alpha < 0:
            raise ValueError('alpha was < 0')
        if alpha > 1:
            raise ValueError('alpha was > 1')
        if alpha < self.alpha_min:
            return self.pos_min
        elif alpha < self.alpha_max:
            delta = ((alpha - self.alpha_min) /
                     (self.alpha_max - self.alpha_min))
            return delta * (self.pos_max - self.pos_min) + self.pos_min
        else:
            return self.pos_max


GMMParams = namedtuple('GMMParams', ['n_components', 'n_distances', 'atoms',
                                     'weights', 'means', 'precisions'])
