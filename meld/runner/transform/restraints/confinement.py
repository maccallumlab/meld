#
# All rights reserved
#

"""
This module implements transformers that add confinement restraints
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


FORCE_GROUP = 3


class ConfinementRestraintTransformer(transform.TransformerBase):
    """
    Transformer to handle confinement restraints
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
            if isinstance(r, restraints.ConfinementRestraint)
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
                confinement_force = mm.CustomExternalForce(
                    "step(r - radius) * force_const * (radius - r)^2;"
                    "r = periodicdistance(x, y, z, 0, 0 ,0)"
                )
            else:
                confinement_force = mm.CustomExternalForce(
                    "step(r - radius) * force_const * (radius - r)^2;"
                    "r=sqrt(x*x + y*y + z*z)"
                )
            confinement_force.addPerParticleParameter("radius")
            confinement_force.addPerParticleParameter("force_const")

            # add the atoms
            for r in self.restraints:
                weight = r.force_const
                confinement_force.addParticle(r.atom_index, [r.radius, weight])
            confinement_force.setForceGroup(FORCE_GROUP)
            system.addForce(confinement_force)
            self.force = confinement_force

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
                    index, r.atom_index, [r.radius, weight]
                )
            self.force.updateParametersInContext(simulation.context)
