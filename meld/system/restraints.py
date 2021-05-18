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

.. note::
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

.. note::
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

from meld import interfaces
from meld.system import indexing

import math
import numpy as np  # type: ignore
from numpy.typing import ArrayLike
from collections import namedtuple
from typing import Dict, Any, Optional, Union, List


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

    def __init__(
        self,
        system: interfaces.ISystem,
        scaler: Optional[RestraintScaler],
        ramp: Optional[TimeRamp],
        atom1: indexing.AtomIndex,
        atom2: indexing.AtomIndex,
        r1: Union[float, Positioner],
        r2: Union[float, Positioner],
        r3: Union[float, Positioner],
        r4: Union[float, Positioner],
        k: Union[float, Positioner],
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
            r1: in nanometers
            r2: in nanometers
            r3: in nanometers
            r4: in nanometers
            k: in :math:`kJ/mol/nm^2`
        """
        assert isinstance(atom1, indexing.AtomIndex)
        self.atom_index_1 = int(atom1)
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

        self.k = k
        self.scaler = scaler
        self.ramp = ramp
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
        weights: ArrayLike,
        means: ArrayLike,
        precisions: ArrayLike,
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
        r1: float,
        r2: float,
        r3: float,
        r4: float,
        k: float,
        asymptote: float,
    ):
        """
        Initialize a HyperbolicDistanceRestraint

        Args:
            system: the system this restraint belongs to
            scaler: scale the force constant with alpha
            ramp: ramp up restraint over time
            atom1: first atom in bond
            atom2: second atom in bond
            r1: distance in nm
            r2: distance in nm
            r3: distance in nm
            r4: distance in nm
            asymptote: maximum energy in kT
        """
        assert isinstance(atom1, indexing.AtomIndex)
        self.atom_index_1 = int(atom1)
        assert isinstance(atom2, indexing.AtomIndex)
        self.atom_index_2 = int(atom2)
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
        phi: float,
        delta_phi: float,
        k: float,
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
        self.phi = phi
        self.delta_phi = delta_phi
        self.k = k
        self.scaler = scaler
        self.ramp = ramp
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
        r_min: float,
        r_max: float,
        n_bins: int,
        spline_params: np.ndarray,
        scale_factor: float,
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
        self.scaler = scaler
        self.ramp = ramp
        assert isinstance(atom1, indexing.AtomIndex)
        assert isinstance(atom2, indexing.AtomIndex)
        self.atom_index_1 = int(atom1)
        self.atom_index_2 = int(atom2)
        self.r_min = r_min
        self.r_max = r_max
        self.n_bins = n_bins
        self.spline_params = spline_params
        self.scale_factor = scale_factor
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
        scale_factor: float,
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
        self.scaler = scaler
        self.ramp = ramp

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
        self.scale_factor = scale_factor
        self._check()

    def _check(self):
        assert self.n_bins > 0
        n_params = self.n_bins * self.n_bins
        assert self.spline_params.shape[0] == n_params
        assert self.spline_params.shape[1] == 16


class RdcRestraint(NonSelectableRestraint):
    """
    Residual Dipolar Coupling Restraint
    """

    _restraint_key_ = "rdc"

    def __init__(
        self,
        system: interfaces.ISystem,
        scaler: Optional[RestraintScaler],
        ramp: Optional[TimeRamp],
        atom1: indexing.AtomIndex,
        atom2: indexing.AtomIndex,
        kappa: float,
        d_obs: float,
        tolerance: float,
        force_const: float,
        quadratic_cut: float,
        weight: float,
        expt_index: int,
        patcher,
    ):
        """
        Initialize an RdcRestraint

        Args:
            system: the system this restraint belongs to
            scaler: scale the force with alpha
            ramp: scale the force over time
            atom1: the first atom in the RDC
            atom2: the second atom in the RDC
            kappa: prefactor for RDC calculation in :math:`Hz / Angstrom^3`
            d_obs: observed dipolar coupling in Hz
            tolerance: calculed couplings within tolerance (in Hz) of d_obs
                will have zero energy and force
            force_const: force constant in :math:`kJ/mol/Hz^2`
            quadratic_cut: force constant becomes linear bond this deviation s^-1
            weight: dimensionless weight to place on this restraint
            expt_index: integer experiment id
            patcher: the :class:`RdcAlignmentPatcher` used to create the alignment tensor

        .. note::
           Typical values for kappa are:

           - 1H - 1H: :math:`-360300 \ Hz / Angstrom^3`
           - 13C - 1H: :math:`-90600 \ Hz / Angstrom^3`
           - 15N - 1H: :math:`36500 \ Hz / Angstrom^3`
        """
        assert isinstance(atom1, indexing.AtomIndex)
        assert isinstance(atom2, indexing.AtomIndex)
        self.atom_index_1 = int(atom1)
        self.atom_index_2 = int(atom2)
        self.s1_index = int(system.atom_index(patcher.resids[expt_index], "S1"))
        self.s2_index = int(system.atom_index(patcher.resids[expt_index], "S2"))
        self.kappa = float(kappa)
        self.d_obs = float(d_obs)
        self.tolerance = float(tolerance)
        self.force_const = float(force_const)
        self.quadratic_cut = quadratic_cut
        self.weight = float(weight)
        self.expt_index = int(expt_index)
        self.scaler = scaler
        self.ramp = ramp
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
        radius: float,
        force_const: float,
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
        self.radius = float(radius)
        self.force_const = float(force_const)
        self.scaler = scaler
        self.ramp = ramp
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
        x: float,
        y: float,
        z: float,
        delta: float,
        force_const: float,
    ):
        """
        Initialize a CartesianRestraint

        Args:
            system: the system this restraint belongs to
            scaler: scale the force with alpha
            ramp: scale the force over time
            atom_index: the atom to restrain
            x: equilibrium x-coordinate, in nm
            y: equilibrium y-coordinate, in nm
            z: equilibrium z-coordinate, in nm
            delta: energy is zero within delta, in nm
            force_const: force constant in :math:`kJ/mol/nm^2`
        """
        assert isinstance(atom_index, indexing.AtomIndex)
        self.atom_index = int(atom_index)
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
        y: float,
        z: float,
        delta: float,
        force_const: float,
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
        self.y = y
        self.z = z
        self.delta = delta
        self.force_const = force_const
        self.scaler = scaler
        self.ramp = ramp
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
        weights: ArrayLike,
        dims: str,
        force_const: float,
        position: ArrayLike,
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
        self.scaler = scaler
        self.ramp = ramp

        self.dims = dims
        self._check_dims()

        self.force_const = force_const
        if self.force_const < 0:
            raise ValueError("force_const cannot be negative")

        self.position = np.array(position, dtype=float)
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
        force_const: float,
        distance: Union[float, Positioner],
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
        self.scaler = scaler
        self.ramp = ramp

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
        self.force_const = force_const
        if self.force_const < 0:
            raise ValueError("force constant cannot be negative")
        if isinstance(distance, Positioner):
            self.positioner = distance
        else:
            if distance < 0.0:
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

        .. note::
           ``restraint_list`` can contain both :class:`RestraintGroup` and
           :class:`SelectableRestraint`. Any :class:`SelectableRestraints`
           will be put into a singleton :class:`RestraintGroup`.
        """
        self._groups: List[RestraintGroup] = []
        if not restraint_list:
            raise RuntimeError(
                "SelectivelyActiveCollection cannot have empty" "restraint list."
            )
        for rest in restraint_list:
            self._add_restraint(rest)

        if num_active < 0:
            raise RuntimeError("num_active must be >= 0.")
        n_rest = len(self._groups)
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

        if num_active < 0:
            raise RuntimeError("num_active must be >= 0.")
        n_rest = len(self._restraints)
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


class ScalerRegistry(type):
    """
    Metaclass that maintains a registry of scaler types.

    All classes that decend from Scaler inherit ScalerRegistry as their
    metaclass. ScalerRegistry will automatically maintain a map between
    the class attribute '_scaler_key_' and all scaler types.

    The function get_constructor_for_key is used to get the class for the
    corresponding key.
    """

    _scaler_registry: Dict[str, type] = {}

    def __init__(cls, name, bases, attrs):
        if name in ["AlphaMapper", "RestraintScaler", "TimeRamp", "Positioner"]:
            pass  # we don't register the base classes
        else:
            try:
                key = attrs["_scaler_key_"]
            except KeyError:
                raise RuntimeError(
                    f"Scaler type {name} subclasses Scaler, but"
                    "does not set _scaler_key_"
                )
            if key in ScalerRegistry._scaler_registry:
                raise RuntimeError(
                    "Trying to register two different classes"
                    f"with _scaler_key_ = {key}."
                )
            ScalerRegistry._scaler_registry[key] = cls

    @classmethod
    def get_constructor_for_key(self, key):
        """Get the constructor for the scaler type matching key."""
        try:
            return ScalerRegistry._scaler_registry[key]
        except KeyError:
            raise RuntimeError(f'Unknown scaler type "{key}".')


class AlphaMapper(metaclass=ScalerRegistry):
    """Base class for all scalers."""

    def __init__(self):
        self._alpha_min = 0.0
        self._alpha_max = 1.0

    def _check_alpha_range(self, alpha):
        if alpha < 0 or alpha > 1:
            raise RuntimeError(f"0 >= alpha >= 1. alpha is {alpha}.")

    def _handle_boundaries(self, alpha):
        if alpha <= self._alpha_min:
            return 1.0
        elif alpha >= self._alpha_max:
            return 0.0
        else:
            return None

    def _check_alpha_min_max(self):
        if (
            self._alpha_min < 0
            or self._alpha_min > 1
            or self._alpha_max < 0
            or self._alpha_max > 1
        ):
            raise RuntimeError(
                "alpha_min and alpha_max must be in range [0, 1]."
                f"alpha_min={self._alpha_min} alpha_max={self._alpha_max}."
            )
        if self._alpha_min >= self._alpha_max:
            raise RuntimeError(
                "alpha_max must be less than alpha_min."
                f"alpha_min={self._alpha_min} alpha_max={self._alpha_max}."
            )


class RestraintScaler(AlphaMapper):
    """Base class for all resraint scaler classes."""


class ConstantScaler(RestraintScaler):
    """This scaler is "always on" and always returns a value of 1.0"."""

    _scaler_key_ = "constant"

    def __call__(self, alpha):
        self._check_alpha_range(alpha)
        return 1.0


class LinearScaler(RestraintScaler):
    """
    This scaler linearly interpolates from alpha_min to alpha_max.
    """

    _scaler_key_ = "linear"

    def __init__(
        self,
        alpha_min: float,
        alpha_max: float,
        strength_at_alpha_min: float = 1.0,
        strength_at_alpha_max: float = STRENGTH_AT_ALPHA_MAX,
    ):
        """
        Initialize a LinearScaler

        Args:
            alpha_min: minimum alpha value
            alpha_max: maximum alpha value
            strength_at_alpha_min: strength when alpha <= alpha_min
            strength_at_alpha_max: strength when alpha >= alpha_max
        """
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
        scale = (1.0 - scale) * (
            self._strength_at_alpha_max - self._strength_at_alpha_min
        ) + self._strength_at_alpha_min
        return scale


class PlateauLinearScaler(RestraintScaler):
    r"""
    A scaler with a plateau shape

    This scaler linearly interpolates between 0 and 1 from alpha_min to
    alpha_one, keeps the value of 1 until alpha_two and then decreases
    linearly until 0 in alpha_max.

        ------   strength alpha_min --> between two and one
      /        \
     /          \ strength alpha_max --> > alpha_max and
                                           below alphamin
    """

    _scaler_key_ = "plateau"

    def __init__(
        self,
        alpha_min: float,
        alpha_one: float,
        alpha_two: float,
        alpha_max: float,
        strength_at_alpha_min: float = 1.0,
        strength_at_alpha_max: float = STRENGTH_AT_ALPHA_MAX,
    ):
        """
        Initialize a PlateauLinearScaler

        Args:
            alpha_min: minimum alpha value
            alpha_one: lower range of plateau
            alpha_two: upper range of plateau
            alpha_max: maximum alpha value
            strength_at_alpha_min: strength when alpha <= alpha_min
            strength_at_alpha_max: strength when alpha >= alpha_max
        """
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
                scale = 1.0 - (self._alpha_one - alpha) / (
                    self._alpha_one - self._alpha_min
                )
                scale = (
                    scale * (self._strength_at_alpha_min - self._strength_at_alpha_max)
                    + self._strength_at_alpha_max
                )

            elif alpha <= self._alpha_two:
                scale = self._strength_at_alpha_min
            elif alpha <= self._alpha_max:
                # Increasing
                scale = 1.0 - (alpha - self._alpha_two) / (
                    self._alpha_max - self._alpha_two
                )
                scale = (1.0 - scale) * (
                    self._strength_at_alpha_max - self._strength_at_alpha_min
                ) + self._strength_at_alpha_min
            else:
                scale = self._strength_at_alpha_max
        return scale


class NonLinearScaler(RestraintScaler):
    """
    A restraint scaler with a non-linear shape
    """

    _scaler_key_ = "nonlinear"

    def __init__(
        self,
        alpha_min: float,
        alpha_max: float,
        factor: float,
        strength_at_alpha_min: float = 1.0,
        strength_at_alpha_max: float = STRENGTH_AT_ALPHA_MAX,
    ):
        """
        Intialize a NonLinearScaler

        Args:
            alpha_min: minimum alpha value
            alpha_max: maximum alpha value
            factor: controls the non-linear shape, must be >= 0
            strength_at_alpha_min: strength when alpha <= alpha_min
            strength_at_alpha_max: strength when alpha >= alpha_max
        """
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._strength_at_alpha_min = strength_at_alpha_min
        self._strength_at_alpha_max = strength_at_alpha_max
        self._check_alpha_min_max()
        if factor < 1:
            raise RuntimeError(f"factor must be >= 1. factor={factor}.")
        self._factor = factor

    def __call__(self, alpha):
        self._check_alpha_range(alpha)
        scale = self._handle_boundaries(alpha)
        if scale is None:
            delta = (alpha - self._alpha_min) / (self._alpha_max - self._alpha_min)
            norm = 1.0 / (math.exp(self._factor) - 1.0)
            scale = norm * (math.exp(self._factor * (1.0 - delta)) - 1.0)
        scale = (1.0 - scale) * (
            self._strength_at_alpha_max - self._strength_at_alpha_min
        ) + self._strength_at_alpha_min
        return scale


class PlateauNonLinearScaler(RestraintScaler):
    """
    Nonlinear scaler with a plateau shape

    This scaler linearly interpolates between 0 and 1 from alpha_min
    to alpha_one, keeps the value of 1 until alpha_two and then decreases
    linearly until 0 in alpha_max.
    """

    _scaler_key_ = "plateaunonlinear"

    def __init__(
        self,
        alpha_min: float,
        alpha_one: float,
        alpha_two: float,
        alpha_max: float,
        factor: float,
        strength_at_alpha_min: float = 1.0,
        strength_at_alpha_max: float = STRENGTH_AT_ALPHA_MAX,
    ):
        """
        Initialize a PlateauNonlinearScaler

        Args:
            alpha_min: minimum alpha value
            alpha_one: lower range of plateau
            alpha_two: upper range of plateau
            alpha_max: maximum alpha value
            factor: controls the non-linear shape, must be >= 0
            strength_at_alpha_min: strength when alpha <= alpha_min
            strength_at_alpha_max: strength when alpha >= alpha_max
        """
        self._alpha_min = float(alpha_min)
        self._alpha_one = float(alpha_one)
        self._alpha_two = float(alpha_two)
        self._alpha_max = float(alpha_max)
        self._strength_at_alpha_min = strength_at_alpha_min
        self._strength_at_alpha_max = strength_at_alpha_max
        self._check_alpha_min_max()
        if factor < 1:
            raise RuntimeError(f"factor must be >= 1. factor={factor}.")
        self._factor = factor

    def __call__(self, alpha):
        self._check_alpha_range(alpha)
        if alpha <= self._alpha_min:
            scale = self._strength_at_alpha_max
        else:
            if alpha <= self._alpha_one:
                delta = (alpha - self._alpha_min) / (self._alpha_one - self._alpha_min)
                norm = 1.0 / (math.exp(self._factor) - 1.0)
                scale = norm * (math.exp(self._factor * (1.0 - delta)) - 1.0)
                scale = (1.0 - scale) * (
                    self._strength_at_alpha_min - self._strength_at_alpha_max
                ) + self._strength_at_alpha_max
            elif alpha <= self._alpha_two:
                scale = self._strength_at_alpha_min
            elif alpha <= self._alpha_max:
                delta = (alpha - self._alpha_two) / (self._alpha_max - self._alpha_two)
                norm = 1.0 / (math.exp(self._factor) - 1.0)
                scale = norm * (math.exp(self._factor * (1.0 - delta)) - 1.0)
                scale = (1.0 - scale) * (
                    self._strength_at_alpha_max - self._strength_at_alpha_min
                ) + self._strength_at_alpha_min
            else:
                scale = self._strength_at_alpha_max

        return scale


class PlateauSmoothScaler(RestraintScaler):
    """
    A scaler with a plateau shape

    This scaler linearly interpolates between 0 and 1 from alpha_min
    to alpha_one, keeps the value of 1 until alpha_two and then decreases
    linearly until 0 in alpha_max.
    """

    _scaler_key_ = "plateausmooth"

    def __init__(
        self,
        alpha_min: float,
        alpha_one: float,
        alpha_two: float,
        alpha_max: float,
        strength_at_alpha_min: float = 1.0,
        strength_at_alpha_max: float = STRENGTH_AT_ALPHA_MAX,
    ):
        """
        Initialize a PlateauSmoothScaler

        Args:
            alpha_min: minimum alpha value
            alpha_one: lower range of plateau
            alpha_two: upper range of plateau
            alpha_max: maximum alpha value
            strength_at_alpha_min: strength when alpha <= alpha_min
            strength_at_alpha_max: strength when alpha >= alpha_max
        """
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
                delta = (alpha - self._alpha_min) / (self._alpha_one - self._alpha_min)
                scale = delta * delta * (3 - 2 * delta)
                scale = (1.0 - scale) * (
                    self._strength_at_alpha_max - self._strength_at_alpha_min
                ) + self._strength_at_alpha_min
            elif alpha <= self._alpha_two:
                scale = self._strength_at_alpha_min
            elif alpha <= self._alpha_max:
                delta = (alpha - self._alpha_two) / (self._alpha_max - self._alpha_two)
                scale = 1 + delta * delta * (2 * delta - 3)
                scale = (1.0 - scale) * (
                    self._strength_at_alpha_max - self._strength_at_alpha_min
                ) + self._strength_at_alpha_min
            else:
                scale = self._strength_at_alpha_max
        return scale


class GeometricScaler(RestraintScaler):
    """
    Scale restraints geometrically
    """

    _scaler_key_ = "geometric"

    def __init__(
        self,
        alpha_min: float,
        alpha_max: float,
        strength_at_alpha_min: float = 1.0,
        strength_at_alpha_max: float = STRENGTH_AT_ALPHA_MAX,
    ):
        """
        Initialize a GeometricScaler

        Args:
            alpha_min: minimum alpha value
            alpha_max: maximum alpha value
            strength_at_alpha_min: strength when alpha <= alpha_min
            strength_at_alpha_max: strength when alpha >= alpha_max
        """
        self._alpha_min = float(alpha_min)
        self._alpha_max = float(alpha_max)
        self._strength_at_alpha_min = float(strength_at_alpha_min)
        self._strength_at_alpha_max = float(strength_at_alpha_max)
        self._delta_alpha = self._alpha_max - self._alpha_min
        self._check_alpha_min_max()

    def __call__(self, alpha):
        self._check_alpha_range(alpha)

        if alpha < 0 or alpha > 1:
            raise RuntimeError("0 <= alpha <=1 1")

        elif alpha <= self._alpha_min:
            return self._strength_at_alpha_min

        elif alpha <= self._alpha_max:
            frac = (alpha - self._alpha_min) / self._delta_alpha
            delta = math.log(self._strength_at_alpha_max) - math.log(
                self._strength_at_alpha_min
            )
            return math.exp(delta * frac + math.log(self._strength_at_alpha_min))

        else:
            return self._strength_at_alpha_max


class TimeRamp(AlphaMapper):
    """Base class for all time ramp classes."""


class ConstantRamp(TimeRamp):
    """TimeRamp that always returns 1.0"""

    _scaler_key_ = "constant_ramp"

    def __call__(self, timestep):
        if timestep < 0:
            raise ValueError("Timestep is < 0.")
        return 1.0


class LinearRamp(TimeRamp):
    """TimeRamp that interpolates linearly"""

    _scaler_key_ = "linear_ramp"

    def __init__(
        self, start_time: float, end_time: float, start_weight: float, end_weight: float
    ):
        """
        Initialize a LinearTimeRamp

        Args:
            start_time: time to start ramping up
            end_time: time to finish ramping up
            start_weight: weight when time <= start_time
            end_weight: weight when time >= end_time
        """
        self.t_start = float(start_time)
        self.t_end = float(end_time)
        self.w_start = float(start_weight)
        self.w_end = float(end_weight)

    def __call__(self, timestep):
        if timestep < 0:
            raise ValueError("Timestep is < 0.")
        if timestep < self.t_start:
            return self.w_start
        elif timestep < self.t_end:
            return self.w_start + (self.w_end - self.w_start) * (
                float(timestep) - self.t_start
            ) / (self.t_end - self.t_start)
        else:
            return self.w_end


class NonLinearRamp(TimeRamp):
    """
    TimeRamp that interpolates non-linearly
    """

    _scaler_key_ = "nonlinear_ramp"

    def __init__(
        self,
        start_time: float,
        end_time: float,
        start_weight: float,
        end_weight: float,
        factor: float,
    ):
        """
        Initialize a NonLinearTimeRamp

        Args:
            start_time: time to start ramping up
            end_time: time to finish ramping up
            start_weight: weight when time <= start_time
            end_weight: weight when time >= end_time
            factor: controls the shape of the non-linear ramp, must be >= 1
        """
        if end_time <= start_time:
            raise ValueError("end_time must be > start_time")
        if factor < 1.0:
            raise ValueError("factor myst be > 1.0")

        self.t_start = float(start_time)
        self.t_end = float(end_time)
        self.w_start = float(start_weight)
        self.w_end = float(end_weight)
        self.factor = float(factor)

    def __call__(self, timestep):
        if timestep < 0:
            raise ValueError("timestep is < 0.")

        if timestep < self.t_start:
            return self.w_start
        elif timestep < self.t_end:
            # we scale differently depending on if we are ramping up or down
            # we change more slowly at lower values and more rapidly at
            # higher values
            #
            # this is for scaling up
            if self.w_end > self.w_start:
                delta = 1.0 - (float(timestep) - self.t_start) / (
                    self.t_end - self.t_start
                )
                norm = 1.0 / (math.exp(self.factor) - 1.0)
                scale = norm * (math.exp(self.factor * (1.0 - delta)) - 1.0)
                return scale * (self.w_end - self.w_start) + self.w_start
            # this is for scaling down
            else:
                delta = (float(timestep) - self.t_start) / (self.t_end - self.t_start)
                norm = 1.0 / (math.exp(self.factor) - 1.0)
                scale = norm * (math.exp(self.factor * (1.0 - delta)) - 1.0)
                return (1.0 - scale) * (self.w_end - self.w_start) + self.w_start
        else:
            return self.w_end


class TimeRampSwitcher(TimeRamp):
    """
    Switches between two TimeRamp objects.

    Class first_ramp before switching time. At the switching
    time it switches to second_ramp, which it uses thereafter.
    """

    _scaler_key_ = "ramp_switcher"

    def __init__(
        self, first_ramp: TimeRamp, second_ramp: TimeRamp, switching_time: float
    ):
        """
        Initialize a TimeRampSwitcher

        Args:
            first_ramp: active when time < switching_time
            second_ramp: active when time >= switching_time
            switching_time: when to switch
        """
        self.first_ramp = first_ramp
        self.second_ramp = second_ramp
        self.switching_time = switching_time

    def __call__(self, timestep):
        if timestep < self.switching_time:
            return self.first_ramp(timestep)
        else:
            return self.second_ramp(timestep)


class Positioner(AlphaMapper):
    """Base class for all positioner classes."""


class ConstantPositioner(Positioner):
    """Always returns the supplied value."""

    _scaler_key_ = "constant_positioner"

    def __init__(self, value):
        self._value = value

    def __call__(self, alpha):
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        if alpha > 1:
            raise ValueError("alpha must be <= 1")

        return self._value


class LinearPositioner(Positioner):
    """
    Position restraints linearly within a range
    """

    _scaler_key_ = "linear_positioner"

    def __init__(
        self, alpha_min: float, alpha_max: float, pos_min: float, pos_max: float
    ):
        """
        Initialize a LinearPositioner

        Args:
            alpha_min: minimum alpha value
            alpha_max: maximum alpha value
            pos_min: value at alpha_min
            pos_max: value at alpha_max
        """
        if alpha_max <= alpha_min:
            raise ValueError("alpha_max must be > alpha_min")

        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.pos_min = float(pos_min)
        self.pos_max = float(pos_max)

    def __call__(self, alpha):
        if alpha < 0:
            raise ValueError("alpha was < 0")
        if alpha > 1:
            raise ValueError("alpha was > 1")
        if alpha < self.alpha_min:
            return self.pos_min
        elif alpha < self.alpha_max:
            delta = (alpha - self.alpha_min) / (self.alpha_max - self.alpha_min)
            return delta * (self.pos_max - self.pos_min) + self.pos_min
        else:
            return self.pos_max


GMMParams = namedtuple(
    "GMMParams",
    ["n_components", "n_distances", "atoms", "weights", "means", "precisions"],
)
