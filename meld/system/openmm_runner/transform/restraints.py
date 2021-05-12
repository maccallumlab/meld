#
# All rights reserved
#

"""
This module implements transformers that add restraint forces
to the openmm system before simulation
"""

import logging

logger = logging.getLogger(__name__)

from openmm import CustomExternalForce  # type: ignore
from meld.system import restraints
from meld.system.param_sampling import Parameter
from collections import OrderedDict, Callable
from meld.system.openmm_runner.transform import TransformerBase
import openmm as mm  # type: ignore

try:
    from meldplugin import MeldForce  # type: ignore
except ImportError:
    logger.warning(
        "Could not import meldplugin. "
        "Are you sure it is installed correctly?\n"
        "Attempts to use meld restraints will fail."
    )


class ConfinementRestraintTransformer(TransformerBase):
    """
    Transformer to handle confinement restraints

    """

    def __init__(
        self,
        param_manager,
        options,
        always_active_restraints,
        selectively_active_restraints,
    ):
        self.use_pbc = options.solvation == "explicit"
        self.restraints = [
            r
            for r in always_active_restraints
            if isinstance(r, restraints.ConfinementRestraint)
        ]
        _delete_from_always_active(self.restraints, always_active_restraints)

        if self.restraints:
            self.active = True
        else:
            self.active = False

        self.force = None

    def add_interactions(self, state, system, topology):
        if self.active:
            # create the confinement force
            if self.use_pbc:
                confinement_force = CustomExternalForce(
                    "step(r - radius) * force_const * (radius - r)^2;"
                    "r = periodicdistance(x, y, z, 0, 0 ,0)"
                )
            else:
                confinement_force = CustomExternalForce(
                    "step(r - radius) * force_const * (radius - r)^2;"
                    "r=sqrt(x*x + y*y + z*z)"
                )
            confinement_force.addPerParticleParameter("radius")
            confinement_force.addPerParticleParameter("force_const")

            # add the atoms
            for r in self.restraints:
                weight = r.force_const
                confinement_force.addParticle(r.atom_index - 1, [r.radius, weight])
            system.addForce(confinement_force)
            self.force = confinement_force

        return system

    def update(self, state, simulation, alpha, timestep):
        if self.active:
            for index, r in enumerate(self.restraints):
                weight = r.force_const * r.scaler(alpha) * r.ramp(timestep)
                self.force.setParticleParameters(
                    index, r.atom_index - 1, [r.radius, weight]
                )
            self.force.updateParametersInContext(simulation.context)


class RDCRestraintTransformer(TransformerBase):
    def __init__(
        self,
        param_manager,
        options,
        always_active_restraints,
        selectively_active_restraints,
    ):
        self.restraints = [
            r
            for r in always_active_restraints
            if isinstance(r, restraints.RdcRestraint)
        ]
        _delete_from_always_active(self.restraints, always_active_restraints)

        if self.restraints:
            self.active = True
        else:
            self.active = False

        self.force = None

        if self.active:
            # map experiments to restraints
            self.expt_dict = DefaultOrderedDict(list)
            for r in self.restraints:
                self.expt_dict[r.expt_index].append(r)

    def add_interactions(self, state, system, topology):
        # The approach we use is based on
        # Habeck, Nilges, Rieping, J. Biomol. NMR., 2007, 135-144.
        #
        # Rather than solving for the exact alignment tensor
        # every step, we sample from a distribution of alignment
        # tensors.
        #
        # We encode the five components of the alignment tensor in
        # the positions of two dummy atoms relative to the center
        # of mass. The value of kappa should be scaled so that the
        # components of the alignment tensor are approximately unity.
        #
        # There is a restraint on the z-component of the seocnd dummy
        # particle to ensure that it does not diffuse off to ininity,
        # which could cause precision issues.
        if self.active:
            rdc_force = mm.CustomCentroidBondForce(
                5,
                "Erest + z_scaler*Ez;"
                "Erest = (1 - step(dev - quadcut)) * quad + step(dev - quadcut) * linear;"
                "linear = 0.5 * k_rdc * quadcut^2 + k_rdc * quadcut * (dev - quadcut);"
                "quad = 0.5 * k_rdc * dev^2;"
                "dev = max(0, abs(d_obs - dcalc) - flat);"
                "dcalc=2/3 * kappa_rdc/r^5 * (s1*(rx^2-ry^2) + s2*(3*rz^2-r^2) + s3*2*rx*ry + s4*2*rx*rz + s5*2*ry*rz);"
                "r=distance(g4, g5);"
                "rx=x4-x5;"
                "ry=y4-y5;"
                "rz=z4-z5;"
                "s1=x2-x1;"
                "s2=y2-y1;"
                "s3=z2-z1;"
                "s4=x3-x1;"
                "s5=y3-y1;"
                "Ez=(z3-z1)^2;",
            )
            rdc_force.addPerBondParameter("d_obs")
            rdc_force.addPerBondParameter("kappa_rdc")
            rdc_force.addPerBondParameter("k_rdc")
            rdc_force.addPerBondParameter("flat")
            rdc_force.addPerBondParameter("quadcut")
            rdc_force.addPerBondParameter("z_scaler")

            for experiment in self.expt_dict:
                # find the set of all atoms involved in this experiment
                com_ind = set()
                for r in self.expt_dict[experiment]:
                    com_ind.add(r.atom_index_1 - 1)
                    com_ind.add(r.atom_index_2 - 1)
                com_ind = list(com_ind)

                # add groups for the COM and dummy particles
                s1 = self.expt_dict[experiment][0].s1_index - 1
                s2 = self.expt_dict[experiment][0].s2_index - 1
                g1 = rdc_force.addGroup(com_ind)
                g2 = rdc_force.addGroup([s1])
                g3 = rdc_force.addGroup([s2])

                # add non-bonded exclusions between dummy particles and all other atoms
                nb_forces = [
                    f
                    for f in system.getForces()
                    if isinstance(f, mm.NonbondedForce)
                    or isinstance(f, mm.CustomNonbondedForce)
                ]
                for nb_force in nb_forces:
                    n_parts = nb_force.getNumParticles()
                    for i in range(n_parts):
                        if isinstance(nb_force, mm.NonbondedForce):
                            if i != s1:
                                nb_force.addException(
                                    i, s1, 0.0, 0.0, 0.0, replace=True
                                )
                            if i != s2:
                                nb_force.addException(
                                    i, s2, 0.0, 0.0, 0.0, replace=True
                                )
                        else:
                            if i != s1:
                                nb_force.addExclusion(i, s1)
                            if i != s2:
                                nb_force.addExclusion(i, s2)

                for r in self.expt_dict[experiment]:
                    # add groups for the atoms involved in the RDC
                    g4 = rdc_force.addGroup([r.atom_index_1 - 1])
                    g5 = rdc_force.addGroup([r.atom_index_2 - 1])
                    rdc_force.addBond(
                        [g1, g2, g3, g4, g5],
                        [
                            r.d_obs,
                            r.kappa,
                            0.0,
                            r.tolerance,
                            r.quadratic_cut,
                            0,
                        ],  # z_scaler initial value shouldn't matter
                    )

            system.addForce(rdc_force)
            self.force = rdc_force
        return system

    def update(self, state, simulation, alpha, timestep):
        if self.active:
            index = 0
            for experiment in self.expt_dict:
                rests = self.expt_dict[experiment]
                for r in rests:
                    scale = r.scaler(alpha) * r.ramp(timestep)
                    groups, params = self.force.getBondParameters(index)
                    assert params[0] == r.d_obs
                    self.force.setBondParameters(
                        index,
                        groups,
                        [
                            r.d_obs,
                            r.kappa,
                            scale * r.force_const,
                            r.tolerance,
                            r.quadratic_cut,
                            r.ramp(timestep),  # set z_scaler to value of ramp
                        ],
                    )
                    index = index + 1
            self.force.updateParametersInContext(simulation.context)


class CartesianRestraintTransformer(TransformerBase):
    def __init__(
        self,
        param_manager,
        options,
        always_active_restraints,
        selectively_active_restraints,
    ):
        self.use_pbc = options.solvation == "explicit"
        self.restraints = [
            r
            for r in always_active_restraints
            if isinstance(r, restraints.CartesianRestraint)
        ]
        _delete_from_always_active(self.restraints, always_active_restraints)

        if self.restraints:
            self.active = True
        else:
            self.active = False

        self.force = None

    def add_interactions(self, state, system, topology):
        if self.active:
            if self.use_pbc:
                cartesian_force = CustomExternalForce(
                    "0.5 * cart_force_const * r_eff^2;"
                    "r_eff = max(0.0, r - cart_delta);"
                    "r = periodicdistance(x, y, z, cart_x, cart_y, cart_z)"
                )
            else:
                cartesian_force = CustomExternalForce(
                    "0.5 * cart_force_const * r_eff^2;"
                    "r_eff = max(0.0, r - cart_delta);"
                    "r = sqrt(dx*dx + dy*dy + dz*dz);"
                    "dx = x - cart_x;"
                    "dy = y - cart_y;"
                    "dz = z - cart_z;"
                )
            cartesian_force.addPerParticleParameter("cart_x")
            cartesian_force.addPerParticleParameter("cart_y")
            cartesian_force.addPerParticleParameter("cart_z")
            cartesian_force.addPerParticleParameter("cart_delta")
            cartesian_force.addPerParticleParameter("cart_force_const")

            # add the atoms
            for r in self.restraints:
                weight = r.force_const
                cartesian_force.addParticle(
                    r.atom_index - 1, [r.x, r.y, r.z, r.delta, weight]
                )
            system.addForce(cartesian_force)
            self.force = cartesian_force
        return system

    def update(self, state, simulation, alpha, timestep):
        if self.active:
            for index, r in enumerate(self.restraints):
                weight = r.force_const * r.scaler(alpha) * r.ramp(timestep)
                self.force.setParticleParameters(
                    index, r.atom_index - 1, [r.x, r.y, r.z, r.delta, weight]
                )
            self.force.updateParametersInContext(simulation.context)


class YZCartesianTransformer(TransformerBase):
    def __init__(
        self,
        param_manager,
        options,
        always_active_restraints,
        selectively_active_restraints,
    ):
        self.use_pbc = options.solvation == "explicit"
        self.restraints = [
            r
            for r in always_active_restraints
            if isinstance(r, restraints.YZCartesianRestraint)
        ]
        _delete_from_always_active(self.restraints, always_active_restraints)

        if self.restraints:
            self.active = True
        else:
            self.active = False

        self.force = None

    def add_interactions(self, state, system, topology):
        if self.active:
            # create the confinement force
            if self.use_pbc:
                cartesian_force = CustomExternalForce(
                    "0.5 * cart_force_const * r_eff^2;"
                    "r_eff = max(0.0, r - cart_delta);"
                    "r = periodicdistance(0, y, z, 0, cart_y, cart_z);"
                )
            else:
                cartesian_force = CustomExternalForce(
                    "0.5 * cart_force_const * r_eff^2;"
                    "r_eff = max(0.0, r - cart_delta);"
                    "r = sqrt(r2);"
                    "r2 = dy*dy + dz*dz;"
                    "dy = y - cart_y;"
                    "dz = z - cart_z;"
                )
            cartesian_force.addPerParticleParameter("cart_y")
            cartesian_force.addPerParticleParameter("cart_z")
            cartesian_force.addPerParticleParameter("cart_delta")
            cartesian_force.addPerParticleParameter("cart_force_const")

            # add the atoms
            for r in self.restraints:
                weight = r.force_const
                cartesian_force.addParticle(
                    r.atom_index - 1, [r.y, r.z, r.delta, weight]
                )
            system.addForce(cartesian_force)
            self.force = cartesian_force
        return system

    def update(self, state, simulation, alpha, timestep):
        if self.active:
            for index, r in enumerate(self.restraints):
                weight = r.force_const * r.scaler(alpha) * r.ramp(timestep)
                self.force.setParticleParameters(
                    index, r.atom_index - 1, [r.y, r.z, r.delta, weight]
                )
            self.force.updateParametersInContext(simulation.context)


class COMRestraintTransformer(TransformerBase):
    def __init__(
        self,
        param_manager,
        options,
        always_active_restraints,
        selectively_active_restraints,
    ):
        self.restraints = [
            r
            for r in always_active_restraints
            if isinstance(r, restraints.COMRestraint)
        ]
        _delete_from_always_active(self.restraints, always_active_restraints)

        if len(self.restraints) > 1:
            raise RuntimeError("Cannot have more than one COMRestraint")

        if self.restraints:
            self.active = True
        else:
            self.active = False

        self.force = None

    def add_interactions(self, state, system, topology):
        if self.active:
            rest = self.restraints[0]
            # convert indices from 1-based to 0-based
            rest_indices1 = [r - 1 for r in rest.indices1]
            rest_indices2 = [r - 1 for r in rest.indices2]

            # create the expression for the energy
            components = []
            if "x" in rest.dims:
                components.append("(x1-x2)*(x1-x2)")
            if "y" in rest.dims:
                components.append("(y1-y2)*(y1-y2)")
            if "z" in rest.dims:
                components.append("(z1-z2)*(z1-z2)")
            dist_expr = "dist = sqrt({});".format(" + ".join(components))
            energy_expr = "0.5 * com_k * (dist - com_ref_dist)*(dist-com_ref_dist);"
            expr = "\n".join([energy_expr, dist_expr])

            # create the force
            force = mm.CustomCentroidBondForce(2, expr)
            force.addPerBondParameter("com_k")
            force.addPerBondParameter("com_ref_dist")

            # create the restraint with parameters
            if rest.weights1:
                g1 = force.addGroup(rest_indices1, rest.weights1)
            else:
                g1 = force.addGroup(rest_indices1)
            if rest.weights2:
                g2 = force.addGroup(rest_indices2, rest.weights2)
            else:
                g2 = force.addGroup(rest_indices2)
            force_const = rest.force_const
            pos = rest.positioner(0)
            force.addBond([g1, g2], [force_const, pos])

            system.addForce(force)
            self.force = force
        return system

    def update(self, state, simulation, alpha, timestep):
        if self.active:
            rest = self.restraints[0]
            weight = rest.force_const * rest.scaler(alpha) * rest.ramp(timestep)
            position = rest.positioner(alpha)
            groups, _ = self.force.getBondParameters(0)
            self.force.setBondParameters(0, groups, [weight, position])
            self.force.updateParametersInContext(simulation.context)


class AbsoluteCOMRestraintTransformer(TransformerBase):
    def __init__(
        self,
        param_manager,
        options,
        always_active_restraints,
        selectively_active_restraints,
    ):
        self.restraints = [
            r
            for r in always_active_restraints
            if isinstance(r, restraints.AbsoluteCOMRestraint)
        ]
        _delete_from_always_active(self.restraints, always_active_restraints)

        if len(self.restraints) > 1:
            raise RuntimeError("Cannot have more than one AbsoluteCOMRestraint")

        if self.restraints:
            self.active = True
        else:
            self.active = False

        self.force = None

    def add_interactions(self, state, system, topology):
        if self.active:
            rest = self.restraints[0]
            # convert indices from 1-based to 0-based
            indices = [r - 1 for r in rest.indices]

            # create the expression for the energy
            components = []
            if "x" in rest.dims:
                components.append("(x1-abscom_x)*(x1-abscom_x)")
            if "y" in rest.dims:
                components.append("(y1-abscom_y)*(y1-abscom_y)")
            if "z" in rest.dims:
                components.append("(z1-abscom_z)*(z1-abscom_z)")
            dist_expr = "dist2={};".format(" + ".join(components))
            energy_expr = "0.5 * com_k * dist2;"
            expr = "\n".join([energy_expr, dist_expr])

            # create the force
            force = mm.CustomCentroidBondForce(1, expr)
            force.addPerBondParameter("com_k")
            force.addPerBondParameter("abscom_x")
            force.addPerBondParameter("abscom_y")
            force.addPerBondParameter("abscom_z")

            # create the restraint with parameters
            if rest.weights:
                g1 = force.addGroup(indices, rest.weights)
            else:
                g1 = force.addGroup(indices)
            force_const = rest.force_const
            pos_x = rest.position[0]
            pos_y = rest.position[1]
            pos_z = rest.position[2]
            force.addBond([g1], [force_const, pos_x, pos_y, pos_z])

            system.addForce(force)
            self.force = force
        return system

    def update(self, state, simulation, alpha, timestep):
        if self.active:
            rest = self.restraints[0]
            weight = rest.force_const * rest.scaler(alpha) * rest.ramp(timestep)
            pos_x = rest.position[0]
            pos_y = rest.position[1]
            pos_z = rest.position[2]
            groups, _ = self.force.getBondParameters(0)
            self.force.setBondParameters(0, groups, [weight, pos_x, pos_y, pos_z])
            self.force.updateParametersInContext(simulation.context)


class MeldRestraintTransformer(TransformerBase):
    def __init__(
        self,
        param_manager,
        options,
        always_active_restraints,
        selectively_active_restraints,
    ):
        # We use the param_manager to update parameters that can be sampled over.
        self.param_manager = param_manager

        # We need to track the index of the first group and first collection
        # that could potentially need their num_active updated.
        self.first_selective_group = 0
        self.first_selective_collection = 0

        # Extract all of the always-on restraints that need to be handled
        # with the MELD plugin.
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

        self.force = None

    def add_interactions(self, state, system, topology):
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

    def update(self, state, simulation, alpha, timestep):
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
        if isinstance(value, Parameter):
            return int(self.param_manager.extract_value(value, state.parameters))
        else:
            return value


def _add_meld_restraint(rest, meld_force, alpha, timestep):
    scale = rest.scaler(alpha) * rest.ramp(timestep)

    if isinstance(rest, restraints.DistanceRestraint):
        rest_index = meld_force.addDistanceRestraint(
            rest.atom_index_1 - 1,
            rest.atom_index_2 - 1,
            rest.r1(alpha),
            rest.r2(alpha),
            rest.r3(alpha),
            rest.r4(alpha),
            rest.k * scale,
        )

    elif isinstance(rest, restraints.HyperbolicDistanceRestraint):
        rest_index = meld_force.addHyperbolicDistanceRestraint(
            rest.atom_index_1 - 1,
            rest.atom_index_2 - 1,
            rest.r1,
            rest.r2,
            rest.r3,
            rest.r4,
            rest.k * scale,
            rest.asymptote * scale,
        )

    elif isinstance(rest, restraints.TorsionRestraint):
        rest_index = meld_force.addTorsionRestraint(
            rest.atom_index_1 - 1,
            rest.atom_index_2 - 1,
            rest.atom_index_3 - 1,
            rest.atom_index_4 - 1,
            rest.phi,
            rest.delta_phi,
            rest.k * scale,
        )

    elif isinstance(rest, restraints.DistProfileRestraint):
        rest_index = meld_force.addDistProfileRestraint(
            rest.atom_index_1 - 1,
            rest.atom_index_2 - 1,
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
            rest.atom_index_1 - 1,
            rest.atom_index_2 - 1,
            rest.atom_index_3 - 1,
            rest.atom_index_4 - 1,
            rest.atom_index_5 - 1,
            rest.atom_index_6 - 1,
            rest.atom_index_7 - 1,
            rest.atom_index_8 - 1,
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
        a = [a - 1 for a in rest.atoms]
        w = rest.weights
        m = list(rest.means.flatten())

        d, o = _setup_precisions(rest.precisions, nd, nc)
        rest_index = meld_force.addGMMRestraint(nd, nc, scale, a, w, m, d, o)

    else:
        raise RuntimeError(f"Do not know how to handle restraint {rest}")

    return rest_index


def _update_meld_restraint(
    rest,
    meld_force,
    alpha,
    timestep,
    dist_index,
    hyper_index,
    tors_index,
    dist_prof_index,
    tors_prof_index,
    gmm_index,
):
    scale = rest.scaler(alpha) * rest.ramp(timestep)

    if isinstance(rest, restraints.DistanceRestraint):
        meld_force.modifyDistanceRestraint(
            dist_index,
            rest.atom_index_1 - 1,
            rest.atom_index_2 - 1,
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
            rest.atom_index_1 - 1,
            rest.atom_index_2 - 1,
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
            rest.atom_index_1 - 1,
            rest.atom_index_2 - 1,
            rest.atom_index_3 - 1,
            rest.atom_index_4 - 1,
            rest.phi,
            rest.delta_phi,
            rest.k * scale,
        )
        tors_index += 1

    elif isinstance(rest, restraints.DistProfileRestraint):
        meld_force.modifyDistProfileRestraint(
            dist_prof_index,
            rest.atom_index_1 - 1,
            rest.atom_index_2 - 1,
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
            rest.atom_index_1 - 1,
            rest.atom_index_2 - 1,
            rest.atom_index_3 - 1,
            rest.atom_index_4 - 1,
            rest.atom_index_5 - 1,
            rest.atom_index_6 - 1,
            rest.atom_index_7 - 1,
            rest.atom_index_8 - 1,
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
        a = [a - 1 for a in rest.atoms]
        w = rest.weights
        m = list(rest.means.flatten())
        d, o = _setup_precisions(rest.precisions, nd, nc)
        _ = meld_force.modifyGMMRestraint(gmm_index, nd, nc, scale, a, w, m, d, o)
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


def _setup_precisions(precisions, n_distances, n_conditions):
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


def _delete_from_always_active(restraints, always_active):
    for restraint in restraints:
        always_active.remove(restraint)


class DefaultOrderedDict(OrderedDict):
    def __init__(self, default_factory=None, *a, **kw):
        isnone = default_factory is None
        callable = isinstance(default_factory, Callable)
        if not isnone and not callable:
            raise TypeError("first argument must be callable")
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = (self.default_factory,)
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy

        return type(self)(self.default_factory, copy.deepcopy(self.items()))

    def __repr__(self):
        return "OrderedDefaultDict({}, {})".format(
            self.default_factory, OrderedDict.__repr__(self)
        )
