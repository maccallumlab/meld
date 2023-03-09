#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Implements all of the restraints available in MELD.

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

There are two main types of restraints, :class:`SelectableRestraint`
and :class:`NonSelectableRestraint`, which have substantially different
behavior.

:class:`NonSelectableRestraint` are "always on". They may be scaled by
scalers and ramps, but the force from each :class:`NonSelectableRestraint`
is independent of other restraints.

:class:`SelectableRestraint` have forces and energies that depend on
other :class:`SelectableRestraint`. They may be combined into
:class:`RestraintGroup` objects, which allows for the ``n_active`` lowest
energy restraints to be active at each timestep. The remaining
restraints are inactive and do not contribute their forces or
energy to the system for that timestep. This selectable nature
allows for the implmentation of very flexible restraint
strategies useful for a variety of problems in structural
biology [1]_, [2]_.

The standard way to create a restraint is using their
:meth:`RestraintManager.create_restraint` with the appropriate
restraint key:

>>> r = system.restraints.create_restraint(rest_key, params...)


Groups
------
:class:`SelectableRestraint` must be part of a :class:`RestraintGroup`. Each
timestep, the restraints are sorted by energy and the ``num_active``
restraints with the lowest energy are activated for the timestep,
while the rest are ignored. It is not possible to add a
:class:`NonSelectableRestraint` to a :class:`RestraintGroup`.

:class:`RestraintGroups` are created by:

>>> g = system.restraints.create_restraint(list_of_restraints, num_active)

Collections
-----------
There are two types of collection: always on, and selectively active.

Restraints that will always be active are added to a single always on
collection. The standard ways to do this are:

>>> system.restraints.add_as_always_active(restraint)
>>> system.restraints.add_as_always_active_list(list_of_restraints)

Restraints or groups of restraints that will be selected are added
to selectively active collections. A mix of bare
:class:`SelectableRestraint` or :class:`RestraintGroup` objects may be added.
When bare restraints are added, they are automatically placed into
a group containing only that with restraint with ``num_active=1``.
The standard way to create a restraint group is:

>>> system.restraints.add_selectively_active_collection(
        list_of_restraints_and_groups, num_active)

Scalers
-------
Each replica in a MELD simulation has a value ``alpha`` that
runs from ``0.0`` to ``1.0``, inclusive. The lowest replica always
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

Note:
   Despite the name, ramps are created with the :meth:`create_scaler` method.

Positioners
-----------
Positioners are used to control the position or distance in a restraint. They
function similar to Scalers, but rather than returning a value in ``[0, 1]``, they
return a value from a defined range.

Positioners are created and added to a restraint by:

>>> positioner = system.restraints.create_scaler(pos_key, params...)
>>> r = system.restraints.create_restraint(
        rest_key, param=positioner, params...)

Note:
   Despite the name, positioners are created with the ``create_scaler`` method.

Restraint Manager
-----------------
The :class:`System` object maintains a :class:`RestraintManager` object, which is the
primary means for interacting with restraints. Generally, restraints, groups,
scalers, etc are created through the :class:`RestraintManager`, rather than
by direct construction.

References
----------
.. [1] J.L. MacCallum, A. Perez, and K.A. Dill, Determining protein structures
       by combining semireliable data with atomistic physical models by Bayesian
       inference, PNAS, 2015, 112(22), pp.6985--6990.
.. [2] A. Perez, J.L. MacCallum, and K.A. Dill, Accelerating molecular simulations
       of proteins using Bayesian inference on weak information, PNAS, 2015,
       112(38), pp. 11846--11851.
"""


from __future__ import annotations
from meld.system.density import DensityMap

from meld.util import strip_unit
from meld import interfaces
from meld.system import indexing
from meld.system import param_sampling
from meld.system import mapping
from meld.system.scalers import (
    RestraintScaler,
    BlurScaler,
    TimeRamp,
    Positioner,
    ConstantPositioner,
    ConstantScaler,
    ConstantRamp,
    ScalerRegistry,
)
from openmm import unit as u  # type: ignore

import math
import numpy as np  # type: ignore
from typing import Dict, Any, Optional, Union, List, NamedTuple


STRENGTH_AT_ALPHA_MAX = 1e-3  # default strength of restraints at alpha=1.0


class _RestraintRegistry(type):
    """
    Metaclass that maintains a registry of restraint types.

    All classes that descend from Restraint inherit _RestraintRegistry as their
    metaclass. _RestraintRegistry will automatically maintain a map between
    the class attribute '_restraint_key_' and all restraint types.

    The function get_constructor_for_key is used to get the class for the
    corresponding key.
    """

    _restraint_registry: Dict[str, type] = {}

    def __init__(cls, name, bases, attrs):
        if name in ["Restraint", "SelectableRestraint", "NonSelectableRestraint"]:
            pass  # we don't register the base classes
        else:
            try:
                key = attrs["_restraint_key_"]
            except KeyError:
                raise RuntimeError(
                    f"Restraint type {name} subclasses Restraint, "
                    "but does not set _restraint_key_"
                )
            if key in _RestraintRegistry._restraint_registry:
                raise RuntimeError(
                    "Trying to register two different classes"
                    f"with _restraint_key_ = {key}."
                )
            _RestraintRegistry._restraint_registry[key] = cls

    @classmethod
    def get_constructor_for_key(self, key):
        """Get the constructor for the restraint type matching key."""
        try:
            return _RestraintRegistry._restraint_registry[key]
        except KeyError:
            raise RuntimeError(f'Unknown restraint type "{key}".')


class Restraint(metaclass=_RestraintRegistry):
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
    """

    _restraint_key_ = "distance"
    atom_index_1: Union[int, mapping.PeakMapping]
    atom_index_2: Union[int, mapping.PeakMapping]

    def __init__(
        self,
        system: interfaces.ISystem,
        scaler: Optional[RestraintScaler],
        ramp: Optional[TimeRamp],
        atom1: Union[indexing.AtomIndex, mapping.PeakMapping],
        atom2: Union[indexing.AtomIndex, mapping.PeakMapping],
        r1: Union[u.Quantity, Positioner],
        r2: Union[u.Quantity, Positioner],
        r3: Union[u.Quantity, Positioner],
        r4: Union[u.Quantity, Positioner],
        k: u.Quantity,
    ):
        """
        Initialize a DistanceRestraint

        The energy is zero between ``r2`` and ``r3``. It increases
        quadratically between ``r1`` and ``r2`` and between
        ``r3`` and ``r4``. The energy increases linearly below ``r1``
        and above ``r4``.

        Args:
            system: system object that restraint belongs to
            scaler: a Scaler to vary the force constant with alpha.
                If ``None``, then a constant 1.0 scaler will
                be used.
            ramp: a time ramp to turn restraints on a beginning of simulation
            atom_1: index of atom 1
            atom_2: index of atom 2
            r1: distance
            r2: distance
            r3: distance
            r4: distance
            k: force constant
        """
        if isinstance(atom1, mapping.PeakMapping):
            self.atom_index_1 = atom1
        else:
            assert isinstance(atom1, indexing.AtomIndex)
            self.atom_index_1 = int(atom1)
        if isinstance(atom2, mapping.PeakMapping):
            self.atom_index_2 = atom2
        else:
            assert isinstance(atom2, indexing.AtomIndex)
            self.atom_index_2 = int(atom2)

        if isinstance(r1, Positioner):
            self.r1 = r1
        else:
            self.r1 = ConstantPositioner(r1)

        if isinstance(r2, Positioner):
            self.r2 = r2
        else:
            self.r2 = ConstantPositioner(r2)

        if isinstance(r3, Positioner):
            self.r3 = r3
        else:
            self.r3 = ConstantPositioner(r3)

        if isinstance(r4, Positioner):
            self.r4 = r4
        else:
            self.r4 = ConstantPositioner(r4)

        self.k = strip_unit(k, u.kilojoule_per_mole / u.nanometer ** 2)
        self.scaler = ConstantScaler() if scaler is None else scaler
        self.ramp = ConstantRamp() if ramp is None else ramp
        self._check(system)

    def _check(self, system):
        for alpha in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            if (
                self.r1(alpha) < 0
                or self.r2(alpha) < 0
                or self.r3(alpha) < 0
                or self.r4(alpha) < 0
            ):
                raise RuntimeError(
                    "r1 to r4 must be > 0. r1={} r2={} r3={} r4={}.".format(
                        self.r1(alpha), self.r2(alpha), self.r3(alpha), self.r4(alpha)
                    )
                )
            if self.r2(alpha) < self.r1(alpha):
                raise RuntimeError(
                    f"r2 must be >= r1. r1={self.r1(alpha)} r2={self.r2(alpha)}."
                )
            if self.r3(alpha) < self.r2(alpha):
                raise RuntimeError(
                    f"r3 must be >= r2. r2={self.r2(alpha)} r3={self.r3(alpha)}."
                )
            if self.r4(alpha) < self.r3(alpha):
                raise RuntimeError(
                    f"r4 must be >= r3. r3={self.r3(alpha)} r4={self.r4(alpha)}."
                )
        if self.k < 0:
            raise RuntimeError(f"k must be >= 0. k={self.k}.")


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
    """

    _restraint_key_ = "gmm"

    def __init__(
        self,
        system: interfaces.ISystem,
        scaler: Optional[RestraintScaler],
        ramp: Optional[TimeRamp],
        n_distances: int,
        n_components: int,
        atoms: List[indexing.AtomIndex],
        weights: np.ndarray,
        means: np.ndarray,
        precisions: np.ndarray,
    ):
        """
        Initialize a GMMDistanceRestraint

        Args:
            system: system object that restraint belongs to
            scaler: A Scaler to vary the force constant with alpha.
                If ``None``, then a constant 1.0 scaler will
                be used.
            ramp: a time ramp to turn restraints on a beginning of simulation
            n_distances: number of distances involved in GMM; max 32
            n_components: number of mixture components; max 32
            atoms: a lit of length `2 * n_distances`
            weights: the weights for the mixture components, shape(n_components)
            means : the means of each mixture component, shape(n_components, n_distances)
            precisions: the precision (i.e. inverse covariance) of each mixture component,
                    shape(n_components, n_distances, n_distances)
        """
        self.scaler = ConstantScaler() if scaler is None else scaler
        self.ramp = ConstantRamp() if ramp is None else ramp
        self.n_distances = n_distances
        self.n_components = n_components
        self.weights = weights
        self.means = means
        self.precisions = precisions
        self.atoms = None
        self._setup_atoms(atoms, system)
        self._check(system)

    @classmethod
    def from_params(
        cls,
        system: interfaces.ISystem,
        scaler: Optional[RestraintScaler],
        ramp: Optional[TimeRamp],
        params: GMMParams,
    ) -> GMMDistanceRestraint:
        """
        Create a GMMDistanceRestraint from a GMMParams object.

        Args:
            system: system object that restraint belongs to
            scaler: A Scaler to vary the force constant with alpha.
                If ``None``, then a constant 1.0 scaler will
                be used.
            ramp: a time ramp to turn restraints on a beginning of simulation
            params: object to build restraint from
        """
        return cls(
            system,
            scaler,
            ramp,
            params.n_distances,
            params.n_components,
            params.atoms,
            params.weights,
            params.means,
            params.precisions,
        )

    def _setup_atoms(self, pair_list, system):
        self.atoms = []
        for index in pair_list:
            assert isinstance(index, indexing.AtomIndex)
            self.atoms.append(int(index))

    def _check(self, system):
        if len(self.atoms) != 2 * self.n_distances:
            raise RuntimeError("len(atoms) must be 2*n_distances")
        if self.weights.shape[0] != self.n_components:
            raise RuntimeError("weights must have shape (n_components,)")
        if self.means.shape != (self.n_components, self.n_distances):
            raise RuntimeError("means must have shape (n_components, n_distances)")
        if self.precisions.shape != (
            self.n_components,
            self.n_distances,
            self.n_distances,
        ):
            raise RuntimeError(
                "precisions must have shape (n_components, n_distances, n_distances)"
            )
        for i in range(self.n_components):
            if not np.allclose(self.precisions[i, :, :], self.precisions[i, :, :].T):
                raise RuntimeError("precision matrix must be symmetric")
        for i in range(self.n_components):
            # Perform a Cholesky decomposition on each precision matrix.
            # This will fail if the matrix is not positive definite.
            try:
                np.linalg.cholesky(self.precisions[i, :, :])
            except np.linalg.LinAlgError:
                raise RuntimeError("precision matrices must be positive definite")


class HyperbolicDistanceRestraint(SelectableRestraint):
    """
    Hyperbolic distance restraint between two atoms
    """

    _restraint_key_ = "hyperbolic"

    def __init__(
        self,
        system: interfaces.ISystem,
        scaler: Optional[RestraintScaler],
        ramp: Optional[TimeRamp],
        atom1: indexing.AtomIndex,
        atom2: indexing.AtomIndex,
        r1: u.Quantity,
        r2: u.Quantity,
        r3: u.Quantity,
        r4: u.Quantity,
        k: u.Quantity,
        asymptote: u.Quantity,
    ):
        """
        Initialize a HyperbolicDistanceRestraint
        There are five regions::

            I:    r < r1

            II:  r1 < r < r2

            III: r2 < r < r3

            IV:  r3 < r < r4

            V:   r4 < r

        The energy is linear in region I, quadratic in II and IV, and zero in III.

        The energy is hyperbolic in region V, with an asymptotic value set by the
        parameter asymptote. The energy will be 1/3 of the asymptotic value at r=r4.
        The distance between r3 and r4 controls the steepness of the potential.

        Args:
            system: the system this restraint belongs to
            scaler: scale the force constant with alpha
            ramp: ramp up restraint over time
            atom1: first atom in bond
            atom2: second atom in bond
            r1: distance
            r2: distance
            r3: distance
            r4: distance
            asymptote: maximum energy in region V
        """
        self.scaler = ConstantScaler() if scaler is None else scaler
        self.ramp = ConstantRamp() if ramp is None else ramp
        assert isinstance(atom1, indexing.AtomIndex)
        self.atom_index_1 = int(atom1)
        assert isinstance(atom2, indexing.AtomIndex)
        self.atom_index_2 = int(atom2)
        self.r1 = strip_unit(r1, u.nanometer)
        self.r2 = strip_unit(r2, u.nanometer)
        self.r3 = strip_unit(r3, u.nanometer)
        self.r4 = strip_unit(r4, u.nanometer)
        self.k = strip_unit(k, u.kilojoule_per_mole / u.nanometer ** 2)
        self.asymptote = strip_unit(asymptote, u.kilojoule_per_mole)

        self._check(system)

    def _check(self, system):
        if self.r1 < 0 or self.r2 < 0 or self.r3 < 0 or self.r4 < 0:
            raise RuntimeError(
                "r1 to r4 must be > 0. r1={} r2={} r3={} r4={}.".format(
                    self.r1, self.r2, self.r3, self.r4
                )
            )

        if self.r2 < self.r1:
            raise RuntimeError(f"r2 must be >= r1. r1={self.r1} r2={self.r2}.")

        if self.r3 < self.r2:
            raise RuntimeError(f"r3 must be >= r2. r2={self.r2} r3={self.r3}.")

        if self.r4 <= self.r3:
            raise RuntimeError(f"r4 must be > r3. r3={self.r3} r4={self.r4}.")

        if self.k < 0:
            raise RuntimeError(f"k must be >= 0. k={self.k}.")

        if self.asymptote < 0:
            raise RuntimeError(f"asymptote must be >= 0. asymptote={self.asymptote}.")


class TorsionRestraint(SelectableRestraint):
    """
    A Torsion restraint between four atoms
    """

    _restraint_key_ = "torsion"

    def __init__(
        self,
        system: interfaces.ISystem,
        scaler: Optional[RestraintScaler],
        ramp: Optional[TimeRamp],
        atom1: indexing.AtomIndex,
        atom2: indexing.AtomIndex,
        atom3: indexing.AtomIndex,
        atom4: indexing.AtomIndex,
        phi: u.Quantity,
        delta_phi: u.Quantity,
        k: u.Quantity,
    ):
        """
        Initialize a TorsionRestraint

        Args:
            system: the system this restraint belongs to
            scaler: scale the force with alpha
            ramp: ramp up the force over time
            atom1: index of first atom
            atom2: index of second atom
            atom3: index of third atom
            atom4: index of fourth atom
            phi: equilibrium angle in degrees
            delta_phi: flat within delta_phi, degrees
            k: force constant in :math:`kJ/mol/deg^2`
        """
        assert isinstance(atom1, indexing.AtomIndex)
        assert isinstance(atom2, indexing.AtomIndex)
        assert isinstance(atom3, indexing.AtomIndex)
        assert isinstance(atom3, indexing.AtomIndex)
        self.atom_index_1 = int(atom1)
        self.atom_index_2 = int(atom2)
        self.atom_index_3 = int(atom3)
        self.atom_index_4 = int(atom4)
        self.phi = strip_unit(phi, u.degree)
        self.delta_phi = strip_unit(delta_phi, u.degree)
        self.k = strip_unit(k, u.kilojoule_per_mole / u.degree ** 2)
        self.scaler = ConstantScaler() if scaler is None else scaler
        self.ramp = ConstantRamp() if ramp is None else ramp
        self._check()

    def _check(self):
        if (
            len(
                set(
                    [
                        self.atom_index_1,
                        self.atom_index_2,
                        self.atom_index_3,
                        self.atom_index_4,
                    ]
                )
            )
            != 4
        ):
            raise RuntimeError(
                "All four indices of a torsion restraint must be unique."
            )
        if self.phi < -180 or self.phi > 180:
            raise RuntimeError(f"-180 <= phi <= 180. phi was {self.phi}.")
        if self.delta_phi < 0 or self.delta_phi > 180:
            raise RuntimeError(f"0 <= delta_phi < 180. delta_phi was {self.delta_phi}.")
        if self.k < 0:
            raise RuntimeError(f"k >= 0. k was {self.k}.")


class DistProfileRestraint(SelectableRestraint):
    """
    A spline-based distance profile restraint between two atoms
    """

    _restraint_key_ = "dist_prof"

    def __init__(
        self,
        system: interfaces.ISystem,
        scaler: Optional[RestraintScaler],
        ramp: Optional[TimeRamp],
        atom1: indexing.AtomIndex,
        atom2: indexing.AtomIndex,
        r_min: u.Quantity,
        r_max: u.Quantity,
        n_bins: int,
        spline_params: np.ndarray,
        scale_factor: u.Quantity,
    ):
        """
        Initialize a DistProfileRestraint

        Args:
            system: the system this restraint belongs to
            scaler: scale the force with alpha
            ramp; scale the force over time
            atom1: the first atom in the bond
            atom2: the second atom in the bond
            r_min: the minimum distance in the lookup table
            r_max: the maximum distance in the lookup table
            n_bins: the number of bins in the lookup table
            spline_params: the spline coefficient lookup table, shape(n_bins, 4)
            scale_factor: scale the energy
        """
        self.scaler = ConstantScaler() if scaler is None else scaler
        self.ramp = ConstantRamp() if ramp is None else ramp
        assert isinstance(atom1, indexing.AtomIndex)
        assert isinstance(atom2, indexing.AtomIndex)
        self.atom_index_1 = int(atom1)
        self.atom_index_2 = int(atom2)
        self.r_min = strip_unit(r_min, u.nanometer)
        self.r_max = strip_unit(r_max, u.nanometer)
        self.n_bins = n_bins
        self.spline_params = spline_params
        self.scale_factor = strip_unit(scale_factor, u.kilojoule_per_mole)
        self._check()

    def _check(self):
        assert self.r_min >= 0.0
        assert self.r_max > self.r_min
        assert self.n_bins > 0
        assert self.spline_params.shape[0] == self.n_bins
        assert self.spline_params.shape[1] == 4


class TorsProfileRestraint(SelectableRestraint):
    """
    A spline-based restraint between two torsions over eight atoms
    """

    _restraint_key_ = "tors_prof"

    def __init__(
        self,
        system: interfaces.ISystem,
        scaler: Optional[RestraintScaler],
        ramp: Optional[TimeRamp],
        atom1: indexing.AtomIndex,
        atom2: indexing.AtomIndex,
        atom3: indexing.AtomIndex,
        atom4: indexing.AtomIndex,
        atom5: indexing.AtomIndex,
        atom6: indexing.AtomIndex,
        atom7: indexing.AtomIndex,
        atom8: indexing.AtomIndex,
        n_bins: int,
        spline_params: np.ndarray,
        scale_factor: u.Quantity,
    ):
        """
        Initialize a TorsProfileRestraint

        Args:
            system: the system this restraint belongs to
            scaler: scale the force with alpha
            ramp: ramp the strength of the force over time
            atom1: first atom of first torsion
            atom2: second atom of first torsion
            atom3: third atom of first torsion
            atom4: fourth atom of first torsion
            atom5: first atom of second torsion
            atom6: second atom of second torsion
            atom7: third atom of second torsion
            atom8: fourth atom of second torsion
            n_bins: number of bins in lookup
            spline_params: the spline coefficient lookup table, shape(n_bins, 16)
            scale_factor: scale the energy
        """
        self.scaler = ConstantScaler() if scaler is None else scaler
        self.ramp = ConstantRamp() if ramp is None else ramp

        assert isinstance(atom1, indexing.AtomIndex)
        assert isinstance(atom2, indexing.AtomIndex)
        assert isinstance(atom3, indexing.AtomIndex)
        assert isinstance(atom4, indexing.AtomIndex)
        assert isinstance(atom5, indexing.AtomIndex)
        assert isinstance(atom6, indexing.AtomIndex)
        assert isinstance(atom7, indexing.AtomIndex)
        assert isinstance(atom8, indexing.AtomIndex)
        self.atom_index_1 = int(atom1)
        self.atom_index_2 = int(atom2)
        self.atom_index_3 = int(atom3)
        self.atom_index_4 = int(atom4)
        self.atom_index_5 = int(atom5)
        self.atom_index_6 = int(atom6)
        self.atom_index_7 = int(atom7)
        self.atom_index_8 = int(atom8)

        self.n_bins = n_bins
        self.spline_params = spline_params
        self.scale_factor = strip_unit(scale_factor, u.kilojoule_per_mole)
        self._check()

    def _check(self):
        assert self.n_bins > 0
        n_params = self.n_bins * self.n_bins
        assert self.spline_params.shape[0] == n_params
        assert self.spline_params.shape[1] == 16


class RdcRestraint(SelectableRestraint):
    """
    Residual Dipolar Coupling Restraint
    """

    _restraint_key_ = "rdc"
    atom_index_1: Union[int, mapping.PeakMapping]
    atom_index_2: Union[int, mapping.PeakMapping]
    
    def __init__(
        self,
        system: interfaces.ISystem,
        scaler: Optional[RestraintScaler],
        ramp: Optional[TimeRamp],
        atom1: Union[indexing.AtomIndex, mapping.PeakMapping],
        atom2: Union[indexing.AtomIndex, mapping.PeakMapping],
        kappa: u.Quantity,
        d_obs: u.Quantity,
        tolerance: u.Quantity,
        force_const: u.Quantity,
        quadratic_cut: u.Quantity,
        weight: float,
        alignment_index: int,
    ):
        """
        Initialize an RdcRestraint

        Args:
            system: the system this restraint belongs to
            scaler: scale the force with alpha
            ramp: scale the force over time
            atom1: the first atom in the RDC
            atom2: the second atom in the RDC
            kappa: prefactor for RDC calculation in :math:`Hz nm^3`
            d_obs: observed dipolar coupling in :math:`Hz`
            tolerance: calculed couplings within tolerance (in :math:`Hz`) of d_obs
                will have zero energy and force
            force_const: force constant in :math:`kJ/mol/Hz^2`
            quadratic_cut: force constant becomes linear beyond this deviation in :math:`s^-1`
            weight: dimensionless weight to place on this restraint
            alignment_index: which alignment to use

        Note:
           Typical values for kappa are:

           - 1H - 1H: :math:`-360.3 Hz nm^3`
           - 13C - 1H: :math:`-90.6 Hz nm^3`
           - 15N - 1H: :math:`36.5 Hz nm^3`
        """
        if isinstance(atom1, mapping.PeakMapping):
            self.atom_index_1 = atom1
        else:
            assert isinstance(atom1, indexing.AtomIndex)
            self.atom_index_1 = int(atom1)
        if isinstance(atom2, mapping.PeakMapping):
            self.atom_index_2 = atom2
        else:
            assert isinstance(atom2, indexing.AtomIndex)
            self.atom_index_2 = int(atom2)

        kappa = strip_unit(kappa, u.second ** -1 * u.nanometer ** 3)
        d_obs = strip_unit(d_obs, u.second ** -1)
        tolerance = strip_unit(tolerance, u.second ** -1)
        force_const = strip_unit(force_const, u.kilojoule_per_mole * u.second ** 2)
        quadratic_cut = strip_unit(quadratic_cut, u.second ** -1)
        self.alignment_index = alignment_index
        self.kappa = float(kappa)
        self.d_obs = float(d_obs)
        self.tolerance = float(tolerance)
        self.force_const = float(force_const)
        self.quadratic_cut = quadratic_cut
        self.weight = float(weight)
        self.scaler = ConstantScaler() if scaler is None else scaler
        self.ramp = ConstantRamp() if ramp is None else ramp
        self._check(system)

    def _check(self, system):
        if self.atom_index_1 == self.atom_index_2:
            raise ValueError("atom1 and atom2 must be different")
        if self.tolerance < 0:
            raise ValueError("tolerance must be > 0")
        if self.force_const < 0:
            raise ValueError("force_constant must be > 0")
        if self.weight < 0:
            raise ValueError("weight must be > 0")
        if self.quadratic_cut <= 0:
            raise ValueError("quadratic_cut must be > 0")


class ConfinementRestraint(NonSelectableRestraint):
    """
    Confinement restraint from origin

    Confines an atom to be within radius of the origin. These restraints are
    typically set to somewhat larger than the expected radius of gyration of
    the protein and help to keep the structures comapact even when the protein
    is unfolded. Typically used with a :class:`ConstantScaler`.
    """

    _restraint_key_ = "confine"

    def __init__(
        self,
        system: interfaces.ISystem,
        scaler: Optional[RestraintScaler],
        ramp: Optional[TimeRamp],
        atom_index: indexing.AtomIndex,
        radius: u.Quantity,
        force_const: u.Quantity,
    ):
        """
        Initialize a ConfinementRestraint

        Args:
            system: the system that this restraint belongs to
            scaler: scale the force with alpha
            ramp: scale the force over time
            atom_index: the index of the restrained atom
            radius: the distance to confine to
            force_const: strength of confinement
        """
        assert isinstance(atom_index, indexing.AtomIndex)
        self.atom_index = int(atom_index)
        self.radius = strip_unit(radius, u.nanometer)
        self.force_const = strip_unit(
            force_const, u.kilojoule_per_mole / u.nanometer ** 2
        )
        self.scaler = ConstantScaler() if scaler is None else scaler
        self.ramp = ConstantRamp() if ramp is None else ramp
        self._check(system)

    def _check(self, system):
        if self.radius < 0:
            raise ValueError("radius must be > 0")
        if self.force_const < 0:
            raise ValueError("force_constant must be > 0")


class CartesianRestraint(NonSelectableRestraint):
    """Cartesian restraint on xyz coordinates"""

    _restraint_key_ = "cartesian"

    def __init__(
        self,
        system: interfaces.ISystem,
        scaler: Optional[RestraintScaler],
        ramp: Optional[TimeRamp],
        atom_index: indexing.AtomIndex,
        x: u.Quantity,
        y: u.Quantity,
        z: u.Quantity,
        delta: u.Quantity,
        force_const: u.Quantity,
    ):
        """
        Initialize a CartesianRestraint

        Args:
            system: the system this restraint belongs to
            scaler: scale the force with alpha
            ramp: scale the force over time
            atom_index: the atom to restrain
            x: equilibrium x-coordinate
            y: equilibrium y-coordinate
            z: equilibrium z-coordinate
            delta: energy is zero within delta
            force_const: force constant
        """
        assert isinstance(atom_index, indexing.AtomIndex)
        self.atom_index = int(atom_index)
        self.x = strip_unit(x, u.nanometer)
        self.y = strip_unit(y, u.nanometer)
        self.z = strip_unit(z, u.nanometer)
        self.delta = strip_unit(delta, u.nanometer)
        self.force_const = strip_unit(
            force_const, u.kilojoule_per_mole / u.nanometer ** 2
        )
        self.scaler = ConstantScaler() if scaler is None else scaler
        self.ramp = ConstantRamp() if ramp is None else ramp
        self._check()

    def _check(self):
        if self.delta < 0:
            raise ValueError("delta must be non-negative")
        if self.force_const < 0:
            raise ValueError("force_const must be non-negative")


class YZCartesianRestraint(NonSelectableRestraint):
    """
    Cartesian restraint on yz coordinates only
    """

    _restraint_key_ = "yzcartesian"

    def __init__(
        self,
        system: interfaces.ISystem,
        scaler: Optional[RestraintScaler],
        ramp: Optional[TimeRamp],
        atom_index: indexing.AtomIndex,
        y: u.Quantity,
        z: u.Quantity,
        delta: u.Quantity,
        force_const: u.Quantity,
    ):
        """
        Initialize a YZCartesianRestraint

        Args:
            system: the system this restraint belongs to
            scaler: scale the force with alpha
            ramp: scale the force over time
            atom_index: the atom to restrain
            x: equilibrium x-coordinate, in nm
            y: equilibrium y-coordinate, in nm
            delta: energy is zero within delta, in nm
            force_const: force constant in :math:`kJ/mol/nm^2`
        """
        assert isinstance(atom_index, indexing.AtomIndex)
        self.atom_index = int(atom_index)
        self.y = strip_unit(y, u.nanometer)
        self.z = strip_unit(z, u.nanometer)
        self.delta = strip_unit(delta, u.nanometer)
        self.force_const = strip_unit(
            force_const, u.kilojoule_per_mole / u.nanometer ** 2
        )
        self.scaler = ConstantScaler() if scaler is None else scaler
        self.ramp = ConstantRamp() if ramp is None else ramp
        self._check()

    def _check(self):
        if self.delta < 0:
            raise ValueError("delta must be non-negative")
        if self.force_const < 0:
            raise ValueError("force_const must be non-negative")


class AbsoluteCOMRestraint(NonSelectableRestraint):
    """
    Restraint on the distance between a group and a point in space

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
    """

    _restraint_key_ = "abs_com"

    def __init__(
        self,
        system: interfaces.ISystem,
        scaler: Optional[RestraintScaler],
        ramp: Optional[TimeRamp],
        group: List[indexing.AtomIndex],
        weights: np.ndarray,
        dims: str,
        force_const: u.Quantity,
        position: u.Quantity,
    ):
        """
        Initialize an AbsoluteCOMRestraint

        Args:
            system: system object used for indexing
            scaler: scale the force with alpha
            ramp: scale the force over time
            group: atoms to restrain COM
            weights: Weights to use when calculating the COM. If ``None``,
                then the masses will be used.
            dims: combination of x, y, z that determines which dimensions
                are used when calculating the distance
            force_const: force constant in kJ/mol/nm^2
            point: location in space to restrain to
        """
        self.scaler: RestraintScaler = ConstantScaler() if scaler is None else scaler
        self.ramp: TimeRamp = ConstantRamp() if ramp is None else ramp

        self.dims = dims
        self._check_dims()

        self.force_const = strip_unit(
            force_const, u.kilojoule_per_mole / u.nanometer ** 2
        )
        if self.force_const < 0:
            raise ValueError("force_const cannot be negative")

        assert isinstance(position, u.Quantity)
        self.position = position.value_in_unit(u.nanometer)
        if len(self.position) != 3:
            raise ValueError("position should be an array of [x, y, z]")

        self.weights = weights
        self.indices = self._get_indices(group)
        self._check_weights()

    def _check_weights(self):
        if self.weights is not None:
            if len(self.indices) != len(self.weights):
                raise ValueError("weights and group have different lengths")
            for w in self.weights:
                if w < 0:
                    raise ValueError("weights must be > 0")

    def _check_dims(self):
        for c in self.dims:
            if c not in "xyz":
                raise ValueError(f'dims must be a combination of "xyz", found {c}')
        for c in "xyz":
            if self.dims.count(c) > 1:
                raise ValueError(f"{c} occurs more than once in dims")

    def _get_indices(self, group):
        indices = []
        for g in group:
            assert isinstance(g, indexing.AtomIndex)
            indices.append(int(g))
        return indices


class COMRestraint(NonSelectableRestraint):
    """
    Restraint on the distance between two groups along selected axes

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
    """

    _restraint_key_ = "com"

    def __init__(
        self,
        system: interfaces.ISystem,
        scaler: Optional[RestraintScaler],
        ramp: Optional[TimeRamp],
        group1: List[indexing.AtomIndex],
        group2: List[indexing.AtomIndex],
        weights1: List[float],
        weights2: List[float],
        dims: str,
        force_const: u.Quantity,
        distance: Union[u.Quantity, Positioner],
    ):
        """
        Initialize a COMRestraint

        Args:
            system: the system this restraint belongs to
            scaler: scale the force with alpha
            ramp: scale the force over time
            group1: atoms in group1
            group2: atoms in group2
            weights1: Weights to use when calculating the COM. If ``None``,
                then the atom masses will be used.
            weights2: Weights to use when calculating the COM. If ``None``,
                then the atom masses will be used.
            dims: combination of x, y, z that determines which dimensions
                are used when calculating the distance
            force_const: force constant in kJ/mol/nm^2
            distance: distance between groups
        """
        # setup indices
        self.scaler = ConstantScaler() if scaler is None else scaler
        self.ramp = ConstantRamp() if ramp is None else ramp

        self.indices1 = self._get_indices(group1)
        self.indices2 = self._get_indices(group2)

        # setup the weights
        self.weights1 = weights1
        if self.weights1 is not None:
            if len(self.indices1) != len(self.weights1):
                raise ValueError("len(indices1) != len(weights1)")
            for w in self.weights1:
                if w < 0:
                    raise ValueError("weights1 must be > 0")
        self.weights2 = weights2
        if weights2 is not None:
            if len(self.indices2) != len(weights2):
                raise ValueError("len(indices2) != len(weights2)")
            for w in self.weights2:
                if w < 0:
                    raise ValueError("weights2 must be > 0")

        # setup the dimensions
        self.dims = dims
        self._check_dims()

        # setup the force constant and positioner
        self.force_const = strip_unit(
            force_const, u.kilojoule_per_mole / u.nanometer ** 2
        )
        if self.force_const < 0:
            raise ValueError("force constant cannot be negative")
        if isinstance(distance, Positioner):
            self.positioner = distance
        else:
            if strip_unit(distance, u.nanometer) < 0.0:
                raise ValueError("distance cannot be negative")

            self.positioner = ConstantPositioner(distance)

    def _get_indices(self, group):
        indices = []
        for g in group:
            assert isinstance(g, indexing.AtomIndex)
            indices.append(int(g))
        return indices

    def _check_dims(self):
        # check for non 'xyz'
        for c in self.dims:
            if c not in "xyz":
                raise ValueError(f'dims must be a combination of "xyz", found {c}')
        for dim in "xyz":
            count = self.dims.count(dim)
            if count > 1:
                raise ValueError(f"{dim} occurs more than once in dims")


class DensityRestraint(SelectableRestraint):

    _restraint_key_ = "density"

    def __init__(
            self,
            system: interfaces.ISystem,
            scaler: Optional[RestraintScaler],
            ramp: Optional[TimeRamp],
            atom: list,
            density: DensityMap,
            mu = None,
        ):
            self.atom_index = [int(i) for i in atom]
            self.scaler = ConstantScaler() if scaler is None else scaler
            self.ramp = ConstantRamp() if ramp is None else ramp
            self.mu = density.density_data
            self.map_origin = density.origin
            self.map_dimension = [density.nx,density.ny,density.nz]
            self.map_gridLength = density.voxel_size


class AlwaysActiveCollection:
    """
    A collection of restraints that are always on
    """

    def __init__(self):
        self._restraints = []

    @property
    def restraints(self) -> List[Restraint]:
        return self._restraints

    def add_restraint(self, restraint: Restraint):
        """
        Add a restraint

        Args:
            restraint: restraint to add
        """
        if not isinstance(restraint, Restraint):
            raise RuntimeError(
                f"Tried to add unknown restraint of type {str(type(restraint))}."
            )
        self._restraints.append(restraint)


class SelectivelyActiveCollection:
    """
    A collection of :class:`RestraintGroup` that are selectively active

    Each time step the ``num_active`` lowest energy groups will be active.
    """

    def __init__(
        self,
        restraint_list: List[Union[RestraintGroup, SelectableRestraint]],
        num_active: int,
    ):
        """
        Initialize a SelectivelyActiveCollection

        Args
            restraint_list: list of restraints to add to collection
            num_active: number active each time step

        Note:
           ``restraint_list`` can contain both :class:`RestraintGroup` and
           :class:`SelectableRestraint`. Any :class:`SelectableRestraints`
           will be put into a singleton :class:`RestraintGroup`.
        """
        self._groups: List[RestraintGroup] = []
        if not restraint_list:
            raise RuntimeError(
                "SelectivelyActiveCollection cannot have empty restraint list."
            )
        for rest in restraint_list:
            self._add_restraint(rest)

        # Do error checking
        n_rest = len(self._groups)
        if isinstance(num_active, param_sampling.DiscreteParameter):
            if num_active.min < 0:
                raise RuntimeError("num_active must be >= 0.")
            if num_active.max > n_rest:
                raise RuntimeError(f"num active must be <= num_groups ({n_rest}).")
        else:
            if num_active < 0:
                raise RuntimeError("num_active must be >= 0.")
            if num_active > n_rest:
                raise RuntimeError(f"num active must be <= num_groups ({n_rest}).")
        self._num_active = num_active

    @property
    def groups(self) -> List[RestraintGroup]:
        """
        Number of groups in collection
        """
        return self._groups

    @property
    def num_active(self) -> int:
        """
        Number active in collection
        """
        return self._num_active

    def _add_restraint(self, restraint):
        if isinstance(restraint, RestraintGroup):
            self._groups.append(restraint)
        elif not isinstance(restraint, SelectableRestraint):
            raise RuntimeError(
                f"Cannot add restraint of type {str(type(restraint))} to"
                "SelectivelyActiveCollection"
            )
        else:
            group = RestraintGroup([restraint], 1)
            self._groups.append(group)


class RestraintGroup:
    """
    A group of selectable restraints

    Each timestep the lowest ``num_active`` energy restraints will be active.
    """

    def __init__(self, rest_list: List[SelectableRestraint], num_active: int):
        """
        Initialize a RestraintGroup

        Args:
            rest_list: list of :class:`SelectableRestraint` in this group
            num_active: number active each timestep
        """
        self._restraints: List[SelectableRestraint] = []
        if not rest_list:
            raise RuntimeError("rest_list cannot be empty.")
        for rest in rest_list:
            self._add_restraint(rest)

        n_rest = len(self._restraints)
        if isinstance(num_active, param_sampling.DiscreteParameter):
            if num_active.min < 0:
                raise RuntimeError("num_active must be >= 0.")
            if num_active.max > n_rest:
                raise RuntimeError(f"num active must be <= num_restraints ({n_rest}).")
        else:
            if num_active < 0:
                raise RuntimeError("num_active must be >= 0.")
            if num_active > n_rest:
                raise RuntimeError(f"num_active must be <= n_rest ({n_rest}).")
        self._num_active = num_active

    @property
    def restraints(self) -> List[SelectableRestraint]:
        """
        Restraints in the group
        """
        return self._restraints

    @property
    def num_active(self) -> int:
        """
        Number of active restraints
        """
        return self._num_active

    def _add_restraint(self, rest):
        if not isinstance(rest, SelectableRestraint):
            raise RuntimeError("Can only add SelectableRestraints to a RestraintGroup.")
        self._restraints.append(rest)


class RestraintManager:
    """
    A class to manage restraints for a System
    """

    def __init__(self, system: interfaces.ISystem):
        """
        Initialize a RestraintManager

        Args:
            system: the System to manage restraints for
        """
        self._system = system
        self._always_active = AlwaysActiveCollection()
        self._selective_collections: List[SelectivelyActiveCollection] = []

    @property
    def always_active(self) -> List[Restraint]:
        """
        Always active restraints
        """
        return self._always_active.restraints

    @property
    def selectively_active_collections(self) -> List[SelectivelyActiveCollection]:
        """
        Selectively active collections
        """
        return self._selective_collections

    def add_as_always_active(
        self, restraint: Union[NonSelectableRestraint, SelectableRestraint]
    ) -> None:
        """
        Add a restraint as always active

        Args:
            restraint: the restraint to add
        """
        self._always_active.add_restraint(restraint)

    def add_as_always_active_list(
        self, restraint_list: List[Union[NonSelectableRestraint, SelectableRestraint]]
    ) -> None:
        """
        Add a list of restraints as always active

        Args:
            restraint_list: the restraints to add
        """
        for r in restraint_list:
            self.add_as_always_active(r)

    def add_selectively_active_collection(
        self,
        rest_list: List[Union[RestraintGroup, SelectableRestraint]],
        num_active: int,
    ) -> None:
        """
        Add a selectively active collection

        Args:
            rest_list: list of restraints or restraint groups to add
            num_active: number of active groups in collection
        """
        self._selective_collections.append(
            SelectivelyActiveCollection(rest_list, num_active)
        )

    def create_restraint(
        self,
        rest_type: str,
        scaler: Optional[RestraintScaler] = None,
        ramp: Optional[TimeRamp] = None,
        **kwargs,
    ) -> Restraint:
        r"""
        Create a restraint

        Args:
            rest_type: type of restraint to add
            scaler: scale the force with alpha
            ramp: scale the force over time
            \**kwargs: passed along to restraint creation functions
        """
        if scaler is None:
            scaler = ConstantScaler()
        else:
            if not isinstance(scaler, RestraintScaler):
                raise ValueError(
                    "scaler must be a subclass of RestraintScaler, "
                    f"you tried to add a {type(scaler)}."
                )

        if ramp is None:
            ramp = ConstantRamp()
        else:
            if not isinstance(ramp, TimeRamp):
                raise ValueError(
                    "ramp must be a subclass of TimeRamp,"
                    f"you tried to add a {type(ramp)}."
                )

        return _RestraintRegistry.get_constructor_for_key(rest_type)(
            self._system, scaler, ramp, **kwargs
        )

    def create_restraint_group(
        self, rest_list: List[SelectableRestraint], num_active: int
    ) -> RestraintGroup:
        """
        Create a restraint group

        Args:
            rest_list: restraints to include in group
            num_active: number of restraints active at each timestep

        Returns:
            the new restraint group
        """
        return RestraintGroup(rest_list, num_active)

    def create_scaler(self, scaler_type: str, **kwargs) -> RestraintScaler:
        r"""
        Create a restraint scaler

        Args:
            scaler_type: the type a scaler to create
            \**kwargs: passed along to the scaler creattion functions

        Returns:
            the new restraint scaler
        """
        return ScalerRegistry.get_constructor_for_key(scaler_type)(**kwargs)


class GMMParams(NamedTuple):
    n_components: int
    n_distances: int
    atoms: List[indexing.AtomIndex]
    weights: np.ndarray
    means: np.ndarray
    precisions: np.ndarray
