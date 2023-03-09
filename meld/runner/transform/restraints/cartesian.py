#
# All rights reserved
#

"""
This module implements transformers that add cartesian restraints
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


FORCE_GROUP = 5

class CartesianRestraintTransformer(transform.TransformerBase):
    """
    Transformer to handle Cartesian restraints
    """

    force: mm.CustomExternalForce

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
        self.use_pbc = builder_info.get("solvation", "implicit") == "explicit"
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

    def add_interactions(
        self, state: interfaces.IState, system: mm.System, topology: app.Topology
    ) -> mm.System:
        if self.active:
            if self.use_pbc:
                cartesian_force = mm.CustomExternalForce(
                    "0.5 * cart_force_const * r_eff^2;"
                    "r_eff = max(0.0, r - cart_delta);"
                    "r = periodicdistance(x, y, z, cart_x, cart_y, cart_z)"
                )
            else:
                cartesian_force = mm.CustomExternalForce(
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
                    r.atom_index, [r.x, r.y, r.z, r.delta, weight]
                )
            cartesian_force.setForceGroup(FORCE_GROUP)
            system.addForce(cartesian_force)
            self.force = cartesian_force
        return system

    def update(
        self,
        state: interfaces.IState,
        simulation: app.Simulation,
        alpha: float,
        timestep: int,
    ) -> None:
        if self.active:
            for index, r in enumerate(self.restraints):
                weight = r.force_const * r.scaler(alpha) * r.ramp(timestep)
                self.force.setParticleParameters(
                    index, r.atom_index, [r.x, r.y, r.z, r.delta, weight]
                )
            self.force.updateParametersInContext(simulation.context)


class YZCartesianTransformer(transform.TransformerBase):
    """
    Transformer to handle YZCartesian restraints
    """

    force: mm.CustomExternalForce

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
        self.use_pbc = builder_info.get("solvation", "implicit") == "explicit"
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

    def add_interactions(
        self, state: interfaces.IState, system: mm.System, topology: app.Topology
    ) -> mm.System:
        if self.active:
            # create the confinement force
            if self.use_pbc:
                cartesian_force = mm.CustomExternalForce(
                    "0.5 * cart_force_const * r_eff^2;"
                    "r_eff = max(0.0, r - cart_delta);"
                    "r = periodicdistance(0, y, z, 0, cart_y, cart_z);"
                )
            else:
                cartesian_force = mm.CustomExternalForce(
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
                cartesian_force.addParticle(r.atom_index, [r.y, r.z, r.delta, weight])
            cartesian_force.setForceGroup(FORCE_GROUP)
            system.addForce(cartesian_force)
            self.force = cartesian_force
        return system

    def update(
        self,
        state: interfaces.IState,
        simulation: app.Simulation,
        alpha: float,
        timestep: int,
    ) -> None:
        if self.active:
            for index, r in enumerate(self.restraints):
                weight = r.force_const * r.scaler(alpha) * r.ramp(timestep)
                self.force.setParticleParameters(
                    index, r.atom_index, [r.y, r.z, r.delta, weight]
                )
            self.force.updateParametersInContext(simulation.context)
