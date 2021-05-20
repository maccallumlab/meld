#
# All rights reserved
#

"""
This module implements transformers that add meld restraints
"""

import logging

logger = logging.getLogger(__name__)

from meld.runner.transform.restraints.util import _delete_from_always_active
from meld import interfaces
from meld.system import restraints
from meld.system import options
from meld.system import param_sampling
from meld.runner import transform
from meldplugin import MeldForce  # type: ignore

from simtk import openmm as mm  # type: ignore
from simtk.openmm import app  # type: ignore

import numpy as np  # type: ignore
from typing import List, Tuple


class MeldRestraintTransformer(transform.TransformerBase):
    """
    Transformer to handle MELD restraints
    """

    force: MeldForce

    def __init__(
        self,
        param_manager,
        options: options.RunOptions,
        always_active_restraints: List[restraints.Restraint],
        selectively_active_restraints: List[restraints.SelectivelyActiveCollection],
    ) -> None:
        # We use the param_manager to update parameters that can be sampled over.
        self.param_manager = param_manager

        # We need to track the index of the first group and first collection
        # that could potentially need their num_active updated.
        self.first_selective_group = 0
        self.first_selective_collection = 0

        self.always_on = [
            r
            for r in always_active_restraints
            if isinstance(r, restraints.SelectableRestraint)
        ]
        _delete_from_always_active(self.always_on, always_active_restraints)

        # Gather all of the selectively active restraints.
        self.selective_on = [r for r in selectively_active_restraints]
        for r in self.selective_on:
            selectively_active_restraints.remove(r)

        if self.always_on or self.selective_on:
            self.active = True
        else:
            self.active = False

    def add_interactions(
        self, state: interfaces.IState, system: mm.System, topology: app.Topology
    ) -> mm.System:
        if self.active:
            meld_force = MeldForce()

            # Add all of the always-on restraints
            if self.always_on:
                group_list = []
                for rest in self.always_on:
                    rest_index = _add_meld_restraint(rest, meld_force, 0, 0)
                    # Each restraint goes in its own group.
                    group_index = meld_force.addGroup([rest_index], 1)
                    group_list.append(group_index)
                # All of the always-on restraints go in a single collection
                meld_force.addCollection(group_list, len(group_list))
                # We need to track the number of groups and collections
                # that are always on
                self.first_selective_group = len(group_list)
                self.first_selective_collection = 1

            for coll in self.selective_on:
                group_indices = []
                for group in coll.groups:
                    restraint_indices = []
                    for rest in group.restraints:
                        rest_index = _add_meld_restraint(rest, meld_force, 0, 0)
                        restraint_indices.append(rest_index)
                    # Create the group
                    group_num_active = self._handle_num_active(group.num_active, state)
                    group_index = meld_force.addGroup(
                        restraint_indices, group_num_active
                    )
                    group_indices.append(group_index)
                # Create the collection
                coll_num_active = self._handle_num_active(group.num_active, state)
                meld_force.addCollection(group_indices, coll_num_active)

            system.addForce(meld_force)
            self.force = meld_force
        return system

    def update(
        self,
        state: interfaces.IState,
        simulation: app.Simulation,
        alpha: float,
        timestep: int,
    ) -> None:
        if self.active:
            self._update_restraints(state, simulation, alpha, timestep)
            self._update_groups_collections(state, simulation, alpha, timestep)
            self.force.updateParametersInContext(simulation.context)

    def _update_groups_collections(self, state, simulation, alpha, timestep):
        # Keep track of which group to modify
        group_index = self.first_selective_group

        for i, coll in enumerate(self.selective_on):
            num_active = self._handle_num_active(coll.num_active, state)
            self.force.modifyCollectionNumActive(
                i + self.first_selective_collection, num_active
            )
            for group in coll.groups:
                num_active_group = self._handle_num_active(group.num_active, state)
                self.force.modifyGroupNumActive(group_index, num_active_group)
                group_index += 1

    def _update_restraints(self, state, simulation, alpha, timestep):
        dist_index = 0
        hyper_index = 0
        tors_index = 0
        dist_prof_index = 0
        tors_prof_index = 0
        gmm_index = 0
        if self.always_on:
            for rest in self.always_on:
                (
                    dist_index,
                    hyper_index,
                    tors_index,
                    dist_prof_index,
                    tors_prof_index,
                    gmm_index,
                ) = _update_meld_restraint(
                    rest,
                    self.force,
                    alpha,
                    timestep,
                    dist_index,
                    hyper_index,
                    tors_index,
                    dist_prof_index,
                    tors_prof_index,
                    gmm_index,
                )
        for coll in self.selective_on:
            for group in coll.groups:
                for rest in group.restraints:
                    (
                        dist_index,
                        hyper_index,
                        tors_index,
                        dist_prof_index,
                        tors_prof_index,
                        gmm_index,
                    ) = _update_meld_restraint(
                        rest,
                        self.force,
                        alpha,
                        timestep,
                        dist_index,
                        hyper_index,
                        tors_index,
                        dist_prof_index,
                        tors_prof_index,
                        gmm_index,
                    )

    def _handle_num_active(self, value, state):
        if isinstance(value, param_sampling.Parameter):
            return int(self.param_manager.extract_value(value, state.parameters))
        else:
            return value


def _add_meld_restraint(
    rest, meld_force: MeldForce, alpha: float, timestep: int
) -> int:
    scale = rest.scaler(alpha) * rest.ramp(timestep)

    if isinstance(rest, restraints.DistanceRestraint):
        rest_index = meld_force.addDistanceRestraint(
            rest.atom_index_1,
            rest.atom_index_2,
            rest.r1(alpha),
            rest.r2(alpha),
            rest.r3(alpha),
            rest.r4(alpha),
            rest.k * scale,
        )

    elif isinstance(rest, restraints.HyperbolicDistanceRestraint):
        rest_index = meld_force.addHyperbolicDistanceRestraint(
            rest.atom_index_1,
            rest.atom_index_2,
            rest.r1,
            rest.r2,
            rest.r3,
            rest.r4,
            rest.k * scale,
            rest.asymptote * scale,
        )

    elif isinstance(rest, restraints.TorsionRestraint):
        rest_index = meld_force.addTorsionRestraint(
            rest.atom_index_1,
            rest.atom_index_2,
            rest.atom_index_3,
            rest.atom_index_4,
            rest.phi,
            rest.delta_phi,
            rest.k * scale,
        )

    elif isinstance(rest, restraints.DistProfileRestraint):
        rest_index = meld_force.addDistProfileRestraint(
            rest.atom_index_1,
            rest.atom_index_2,
            rest.r_min,
            rest.r_max,
            rest.n_bins,
            rest.spline_params[:, 0],
            rest.spline_params[:, 1],
            rest.spline_params[:, 2],
            rest.spline_params[:, 3],
            rest.scale_factor * scale,
        )

    elif isinstance(rest, restraints.TorsProfileRestraint):
        rest_index = meld_force.addTorsProfileRestraint(
            rest.atom_index_1,
            rest.atom_index_2,
            rest.atom_index_3,
            rest.atom_index_4,
            rest.atom_index_5,
            rest.atom_index_6,
            rest.atom_index_7,
            rest.atom_index_8,
            rest.n_bins,
            rest.spline_params[:, 0],
            rest.spline_params[:, 1],
            rest.spline_params[:, 2],
            rest.spline_params[:, 3],
            rest.spline_params[:, 4],
            rest.spline_params[:, 5],
            rest.spline_params[:, 6],
            rest.spline_params[:, 7],
            rest.spline_params[:, 8],
            rest.spline_params[:, 9],
            rest.spline_params[:, 10],
            rest.spline_params[:, 11],
            rest.spline_params[:, 12],
            rest.spline_params[:, 13],
            rest.spline_params[:, 14],
            rest.spline_params[:, 15],
            rest.scale_factor * scale,
        )

    elif isinstance(rest, restraints.GMMDistanceRestraint):
        nd = rest.n_distances
        nc = rest.n_components
        w = rest.weights
        m = list(rest.means.flatten())

        d, o = _setup_precisions(rest.precisions, nd, nc)
        rest_index = meld_force.addGMMRestraint(nd, nc, scale, rest.atoms, w, m, d, o)

    else:
        raise RuntimeError(f"Do not know how to handle restraint {rest}")

    return rest_index


def _update_meld_restraint(
    rest,
    meld_force: MeldForce,
    alpha: float,
    timestep: int,
    dist_index: int,
    hyper_index: int,
    tors_index: int,
    dist_prof_index: int,
    tors_prof_index: int,
    gmm_index: int,
) -> Tuple[int, int, int, int, int, int]:
    scale = rest.scaler(alpha) * rest.ramp(timestep)

    if isinstance(rest, restraints.DistanceRestraint):
        meld_force.modifyDistanceRestraint(
            dist_index,
            rest.atom_index_1,
            rest.atom_index_2,
            rest.r1(alpha),
            rest.r2(alpha),
            rest.r3(alpha),
            rest.r4(alpha),
            rest.k * scale,
        )
        dist_index += 1

    elif isinstance(rest, restraints.HyperbolicDistanceRestraint):
        meld_force.modifyHyperbolicDistanceRestraint(
            hyper_index,
            rest.atom_index_1,
            rest.atom_index_2,
            rest.r1,
            rest.r2,
            rest.r3,
            rest.r4,
            rest.k * scale,
            rest.asymptote * scale,
        )
        hyper_index += 1

    elif isinstance(rest, restraints.TorsionRestraint):
        meld_force.modifyTorsionRestraint(
            tors_index,
            rest.atom_index_1,
            rest.atom_index_2,
            rest.atom_index_3,
            rest.atom_index_4,
            rest.phi,
            rest.delta_phi,
            rest.k * scale,
        )
        tors_index += 1

    elif isinstance(rest, restraints.DistProfileRestraint):
        meld_force.modifyDistProfileRestraint(
            dist_prof_index,
            rest.atom_index_1,
            rest.atom_index_2,
            rest.r_min,
            rest.r_max,
            rest.n_bins,
            rest.spline_params[:, 0],
            rest.spline_params[:, 1],
            rest.spline_params[:, 2],
            rest.spline_params[:, 3],
            rest.scale_factor * scale,
        )
        dist_prof_index += 1

    elif isinstance(rest, restraints.TorsProfileRestraint):
        meld_force.modifyTorsProfileRestraint(
            tors_prof_index,
            rest.atom_index_1,
            rest.atom_index_2,
            rest.atom_index_3,
            rest.atom_index_4,
            rest.atom_index_5,
            rest.atom_index_6,
            rest.atom_index_7,
            rest.atom_index_8,
            rest.n_bins,
            rest.spline_params[:, 0],
            rest.spline_params[:, 1],
            rest.spline_params[:, 2],
            rest.spline_params[:, 3],
            rest.spline_params[:, 4],
            rest.spline_params[:, 5],
            rest.spline_params[:, 6],
            rest.spline_params[:, 7],
            rest.spline_params[:, 8],
            rest.spline_params[:, 9],
            rest.spline_params[:, 10],
            rest.spline_params[:, 11],
            rest.spline_params[:, 12],
            rest.spline_params[:, 13],
            rest.spline_params[:, 14],
            rest.spline_params[:, 15],
            rest.scale_factor * scale,
        )
        tors_prof_index += 1

    elif isinstance(rest, restraints.GMMDistanceRestraint):
        nd = rest.n_distances
        nc = rest.n_components
        w = rest.weights
        m = list(rest.means.flatten())
        d, o = _setup_precisions(rest.precisions, nd, nc)
        _ = meld_force.modifyGMMRestraint(
            gmm_index, nd, nc, scale, rest.atoms, w, m, d, o
        )
        gmm_index += 1

    else:
        raise RuntimeError(f"Do not know how to handle restraint {rest}")

    return (
        dist_index,
        hyper_index,
        tors_index,
        dist_prof_index,
        tors_prof_index,
        gmm_index,
    )


def _setup_precisions(
    precisions: np.ndarray, n_distances: int, n_conditions: int
) -> Tuple[List[float], List[float]]:
    # The normalization of our GMMs will blow up
    # due to division by zero if the precisions
    # are zero, so we clamp this to a very
    # small value.
    diags = []
    for i in range(n_conditions):
        for j in range(n_distances):
            diags.append(precisions[i, j, j])

    off_diags = []
    for i in range(n_conditions):
        for j in range(n_distances):
            for k in range(j + 1, n_distances):
                off_diags.append(precisions[i, j, k])

    return diags, off_diags
