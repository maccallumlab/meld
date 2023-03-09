#
# All rights reserved
#

"""
This module implements transformers that add COM restraints
"""

import logging

logger = logging.getLogger(__name__)

from meld.runner.transform.restraints.util import _delete_from_always_active
from meld import interfaces
from meld.system import restraints
from meld.system import options
from meld.system import param_sampling
from meld.system import mapping
from meld.system import density
from meld.runner import transform

import openmm as mm  # type: ignore
from openmm import app  # type: ignore

from typing import List


FORCE_GROUP = 4


class COMRestraintTransformer(transform.TransformerBase):
    """
    Transformer to handle COM restraints
    """

    force: mm.CustomCentroidBondForce

    def __init__(
        self,
        param_manager: param_sampling.ParameterManager,
        mapper: mapping.PeakMapManager,
        density_manager: density.DensityManager,
        builder_info: dict,
        options: options.RunOptions,
        always_active_restraints: List[restraints.Restraint],
        selectively_active_restraints: List[restraints.SelectivelyActiveCollection],
    ) -> None:
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

    def add_interactions(
        self, state: interfaces.IState, system: mm.System, topology: app.Topology
    ) -> mm.System:
        if self.active:
            rest = self.restraints[0]

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
                g1 = force.addGroup(rest.indices1, rest.weights1)
            else:
                g1 = force.addGroup(rest.indices1)
            if rest.weights2:
                g2 = force.addGroup(rest.indices2, rest.weights2)
            else:
                g2 = force.addGroup(rest.indices2)
            force_const = rest.force_const
            pos = rest.positioner(0)
            force.addBond([g1, g2], [force_const, pos])

            force.setForceGroup(FORCE_GROUP)

            system.addForce(force)
            self.force = force
        return system

    def update(
        self,
        state: interfaces.IState,
        simulation: app.Simulation,
        alpha: float,
        timestep: int,
    ) -> None:
        if self.active:
            rest = self.restraints[0]
            weight = rest.force_const * rest.scaler(alpha) * rest.ramp(timestep)
            position = rest.positioner(alpha)
            groups, _ = self.force.getBondParameters(0)
            self.force.setBondParameters(0, groups, [weight, position])
            self.force.updateParametersInContext(simulation.context)


class AbsoluteCOMRestraintTransformer(transform.TransformerBase):
    """
    Transformer to handle AbsoluteCOM restraints
    """

    force: mm.CustomCentroidBondForce

    def __init__(
        self,
        param_manager: param_sampling.ParameterManager,
        mapper: mapping.PeakMapManager,
        density_manager: density.DensityManager,
        builder_info: dict,
        options: options.RunOptions,
        always_active_restraints: List[restraints.Restraint],
        selectively_active_restraints: List[restraints.SelectivelyActiveCollection],
    ) -> None:
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

    def add_interactions(
        self, state: interfaces.IState, system: mm.System, topology: app.Topology
    ) -> mm.System:
        if self.active:
            rest = self.restraints[0]

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
                g1 = force.addGroup(rest.indices, rest.weights)
            else:
                g1 = force.addGroup(rest.indices)
            force_const = rest.force_const
            pos_x = rest.position[0]
            pos_y = rest.position[1]
            pos_z = rest.position[2]
            force.addBond([g1], [force_const, pos_x, pos_y, pos_z])

            force.setForceGroup(FORCE_GROUP)

            system.addForce(force)
            self.force = force
        return system

    def update(
        self,
        state: interfaces.IState,
        simulation: app.Simulation,
        alpha: float,
        timestep: int,
    ) -> None:
        if self.active:
            rest = self.restraints[0]
            weight = rest.force_const * rest.scaler(alpha) * rest.ramp(timestep)
            pos_x = rest.position[0]
            pos_y = rest.position[1]
            pos_z = rest.position[2]
            groups, _ = self.force.getBondParameters(0)
            self.force.setBondParameters(0, groups, [weight, pos_x, pos_y, pos_z])
            self.force.updateParametersInContext(simulation.context)
