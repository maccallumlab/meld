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
from meld.system import mapping
from meld.runner import transform
from meldplugin import MeldForce  # type: ignore
from meld.runner.transform.restraints.meld.tracker import RestraintTracker

from simtk import openmm as mm  # type: ignore
from simtk.openmm import app  # type: ignore

import numpy as np  # type: ignore
from typing import List, Tuple, Union


class MeldRestraintTransformer(transform.TransformerBase):
    """
    Transformer to handle MELD restraints
    """

    force: MeldForce

    def __init__(
        self,
        param_manager: param_sampling.ParameterManager,
        mapper: mapping.PeakMapManager,
        options: options.RunOptions,
        always_active_restraints: List[restraints.Restraint],
        selectively_active_restraints: List[restraints.SelectivelyActiveCollection],
    ) -> None:
        self.param_manager = param_manager
        self.mapper = mapper

        # Track indices of restraints, groups, and collections so that we can
        # update them.
        self.tracker = RestraintTracker(param_manager, mapper)

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
                    rest_index = self._add_meld_restraint(rest, meld_force, 0, 0, state)
                    # Each restraint goes in its own group.
                    # This group does not depend on parameter sampling,
                    # so we will never need to update it
                    group_index = meld_force.addGroup([rest_index], 1)
                    group_list.append(group_index)

                # All of the always-on restraints go in a single collection
                # This collection does not depend on parameter sampling,
                # so we will never need to update it.
                meld_force.addCollection(group_list, len(group_list))

            # Add the selectively active restraints
            for coll in self.selective_on:
                group_indices = []
                for group in coll.groups:
                    restraint_indices = []
                    for rest in group.restraints:
                        rest_index = self._add_meld_restraint(
                            rest, meld_force, 0, 0, state
                        )
                        restraint_indices.append(rest_index)

                    # Create the group in the meldplugin
                    group_num_active = self._handle_num_active(group.num_active, state)
                    group_index = meld_force.addGroup(
                        restraint_indices, group_num_active
                    )
                    group_indices.append(group_index)

                    # If the group depends on parameter sampling, add it to the tracker
                    # so that it can be updated.
                    if isinstance(group.num_active, param_sampling.Parameter):
                        self.tracker.groups_with_dep.append((group, group_index))

                # Create the collection in the meldplugin
                coll_num_active = self._handle_num_active(coll.num_active, state)
                coll_index = meld_force.addCollection(group_indices, coll_num_active)

                # If the collection depends on parameter sampling, add it to the tracker
                # so that it can be updated.
                if isinstance(coll.num_active, param_sampling.Parameter):
                    self.tracker.collections_with_dep.append((coll, coll_index))

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
            self._update_restraints(alpha, timestep, state)
            self._update_groups_collections(state)
            self.force.updateParametersInContext(simulation.context)

    def _update_groups_collections(
        self,
        state: interfaces.IState,
    ) -> None:
        for coll, index in self.tracker.collections_with_dep:
            num_active = self._handle_num_active(coll.num_active, state)
            self.force.modifyCollectionNumActive(index, num_active)

        for group, index in self.tracker.groups_with_dep:
            num_active = self._handle_num_active(group.num_active, state)
            self.force.modifyGroupNumActive(index, num_active)

    def _update_restraints(
        self, alpha: float, timestep: int, state: interfaces.IState
    ) -> None:
        # Get the list of restraints to update
        self.tracker.update(alpha, timestep, state)
        to_update = self.tracker.get_and_reset_need_update()

        for category, index in to_update:
            if category == "distance":
                dist_rest = self.tracker.distance_restraints[index]
                scale = dist_rest.scaler(alpha) * dist_rest.ramp(timestep)
                j, k = self._handle_mapping(
                    [dist_rest.atom_index_1, dist_rest.atom_index_2], state
                )
                self.force.modifyDistanceRestraint(
                    index,
                    j,
                    k,
                    dist_rest.r1(alpha),
                    dist_rest.r2(alpha),
                    dist_rest.r3(alpha),
                    dist_rest.r4(alpha),
                    dist_rest.k * scale,
                )
            elif category == "hyperbolic_distance":
                hyper_rest = self.tracker.hyperbolic_distance_restraints[index]
                scale = hyper_rest.scaler(alpha) * hyper_rest.ramp(timestep)
                self.force.modifyHyperbolicDistanceRestraint(
                    index,
                    hyper_rest.atom_index_1,
                    hyper_rest.atom_index_2,
                    hyper_rest.r1,
                    hyper_rest.r2,
                    hyper_rest.r3,
                    hyper_rest.r4,
                    hyper_rest.k * scale,
                    hyper_rest.asymptote * scale,
                )
            elif category == "torsion":
                tors_rest = self.tracker.torsion_restraints[index]
                scale = tors_rest.scaler(alpha) * tors_rest.ramp(timestep)
                self.force.modifyTorsionRestraint(
                    index,
                    tors_rest.atom_index_1,
                    tors_rest.atom_index_2,
                    tors_rest.atom_index_3,
                    tors_rest.atom_index_4,
                    tors_rest.phi,
                    tors_rest.delta_phi,
                    tors_rest.k * scale,
                )
            elif category == "dist_profile":
                dist_prof_rest = self.tracker.dist_prof_restraints[index]
                scale = dist_prof_rest.scaler(alpha) * dist_prof_rest.ramp(timestep)
                self.force.modifyDistProfileRestraint(
                    index,
                    dist_prof_rest.atom_index_1,
                    dist_prof_rest.atom_index_2,
                    dist_prof_rest.r_min,
                    dist_prof_rest.r_max,
                    dist_prof_rest.n_bins,
                    dist_prof_rest.spline_params[:, 0],
                    dist_prof_rest.spline_params[:, 1],
                    dist_prof_rest.spline_params[:, 2],
                    dist_prof_rest.spline_params[:, 3],
                    dist_prof_rest.scale_factor * scale,
                )
            elif category == "tors_profile":
                tors_prof_rest = self.tracker.torsion_profile_restraints[index]
                scale = tors_prof_rest.scaler(alpha) * tors_prof_rest.ramp(timestep)
                self.force.modifyTorsProfileRestraint(
                    index,
                    tors_prof_rest.atom_index_1,
                    tors_prof_rest.atom_index_2,
                    tors_prof_rest.atom_index_3,
                    tors_prof_rest.atom_index_4,
                    tors_prof_rest.atom_index_5,
                    tors_prof_rest.atom_index_6,
                    tors_prof_rest.atom_index_7,
                    tors_prof_rest.atom_index_8,
                    tors_prof_rest.n_bins,
                    tors_prof_rest.spline_params[:, 0],
                    tors_prof_rest.spline_params[:, 1],
                    tors_prof_rest.spline_params[:, 2],
                    tors_prof_rest.spline_params[:, 3],
                    tors_prof_rest.spline_params[:, 4],
                    tors_prof_rest.spline_params[:, 5],
                    tors_prof_rest.spline_params[:, 6],
                    tors_prof_rest.spline_params[:, 7],
                    tors_prof_rest.spline_params[:, 8],
                    tors_prof_rest.spline_params[:, 9],
                    tors_prof_rest.spline_params[:, 10],
                    tors_prof_rest.spline_params[:, 11],
                    tors_prof_rest.spline_params[:, 12],
                    tors_prof_rest.spline_params[:, 13],
                    tors_prof_rest.spline_params[:, 14],
                    tors_prof_rest.spline_params[:, 15],
                    tors_prof_rest.scale_factor * scale,
                )
            elif category == "gmm":
                gmm_rest = self.tracker.gmm_restraints[index]
                scale = gmm_rest.scaler(alpha) * gmm_rest.ramp(timestep)
                nd = gmm_rest.n_distances
                nc = gmm_rest.n_components
                w = gmm_rest.weights
                m = list(gmm_rest.means.flatten())
                d, o = _setup_precisions(gmm_rest.precisions, nd, nc)
                self.force.modifyGMMRestraint(
                    index, nd, nc, scale, gmm_rest.atoms, w, m, d, o
                )
            else:
                raise RuntimeError(f"Unknown restraint category {category}")

    def _handle_num_active(self, value, state):
        if isinstance(value, param_sampling.Parameter):
            return int(self.param_manager.extract_value(value, state.parameters))
        else:
            return value

    def _handle_mapping(
        self, values: List[Union[int, mapping.PeakMapping]], state: interfaces.IState
    ) -> List[int]:
        indices: List[int] = []
        for value in values:
            if isinstance(value, mapping.PeakMapping):
                index = self.mapper.extract_value(value, state.mappings)
                if isinstance(index, mapping.NotMapped):
                    index = -1
            else:
                index = value
            indices.append(index)

        # If any of the indices is un-mapped, we set them
        # # all to -1.
        if any(x == -1 for x in indices):
            indices = [-1 for _ in values]

        return indices

    def _add_meld_restraint(
        self,
        rest,
        meld_force: MeldForce,
        alpha: float,
        timestep: int,
        state: interfaces.IState,
    ) -> int:
        scale = rest.scaler(alpha) * rest.ramp(timestep)

        if isinstance(rest, restraints.DistanceRestraint):
            i, j = self._handle_mapping([rest.atom_index_1, rest.atom_index_2], state)
            rest_index = meld_force.addDistanceRestraint(
                i,
                j,
                rest.r1(alpha),
                rest.r2(alpha),
                rest.r3(alpha),
                rest.r4(alpha),
                rest.k * scale,
            )
            self.tracker.add_distance_restraint(rest, alpha, timestep, state)

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
            self.tracker.add_hyperbolic_distance_restraint(rest, alpha, timestep, state)

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
            self.tracker.add_torsion_restraint(rest, alpha, timestep, state)

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
            self.tracker.add_distance_profile_restraint(rest, alpha, timestep, state)

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
            self.tracker.add_torsion_profile_restraint(rest, alpha, timestep, state)

        elif isinstance(rest, restraints.GMMDistanceRestraint):
            nd = rest.n_distances
            nc = rest.n_components
            w = rest.weights
            m = list(rest.means.flatten())

            d, o = _setup_precisions(rest.precisions, nd, nc)
            rest_index = meld_force.addGMMRestraint(
                nd, nc, scale, rest.atoms, w, m, d, o
            )
            self.tracker.add_gmm_distance_restraint(rest, alpha, timestep, state)

        else:
            raise RuntimeError(f"Do not know how to handle restraint {rest}")

        return rest_index


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
