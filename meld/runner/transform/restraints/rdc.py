#
# All rights reserved
#

"""
This module implements transformers that add rdc restraints
"""

from ctypes import alignment
import logging

logger = logging.getLogger(__name__)

from meld.runner.transform.restraints.util import _delete_from_always_active
from meld import interfaces
from meld.system import restraints
from meld.system import options
from meld.system import param_sampling
from meld.system import mapping
from meld.runner import transform

import openmm as mm  # type: ignore
from openmm import app  # type: ignore

from collections import defaultdict
from typing import List, Dict, Union


FORCE_GROUP = 2


class RDCRestraintTransformer(transform.TransformerBase):
    """
    Transformer to handle RDC restraints
    """

    force: mm.CustomCentroidBondForce
    alignment_forces: Dict[int, mm.CustomCompoundBondForce]

    def __init__(
        self,
        param_manager: param_sampling.ParameterManager,
        mapper: mapping.PeakMapManager,
        builder_info: dict,
        options: options.RunOptions,
        always_active_restraints: List[restraints.Restraint],
        selectively_active_restraints: List[restraints.SelectivelyActiveCollection],
    ) -> None:
        self.mapper = mapper
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

        self.alignment_forces = {}
        self.alignment_map = defaultdict(list)

        if self.active:
            self.scale_factor = builder_info["alignment_scale_factor"]
            for r in self.restraints:
                self.alignment_map[r.alignment_index].append(r)

    def add_interactions(
        self, state: interfaces.IState, system: mm.System, topology: app.Topology
    ) -> mm.System:
        # The approach we use is based on
        # Habeck, Nilges, Rieping, J. Biomol. NMR., 2007, 135-144.
        #
        # Rather than solving for the exact alignment tensor
        # every step, we sample from a distribution of alignment
        # tensors.
        #
        if self.active:
            # add in all of the forces
            for alignment in self.alignment_map:
                force = _create_rdc_force(alignment, self.scale_factor)
                self.alignment_forces[alignment] = force

            # add each restraint to the appropriate force
            for alignment in self.alignment_map:
                rests = self.alignment_map[alignment]
                force = self.alignment_forces[alignment]
                for r in rests:
                    i, j = self._handle_mapping([r.atom_index_1, r.atom_index_2], state)
                    force.addBond(
                        [i, j],
                        [
                            r.d_obs,
                            r.kappa,
                            r.force_const,
                            r.tolerance,
                            r.quadratic_cut,
                            r.weight,
                        ],
                    )

            # add forces to the system
            for alignment in self.alignment_forces:
                system.addForce(self.alignment_forces[alignment])

        return system

    def update(
        self,
        state: interfaces.IState,
        simulation: app.Simulation,
        alpha: float,
        timestep: int,
    ) -> None:
        if self.active:
            for alignment in self.alignment_map:
                rests = self.alignment_map[alignment]
                force = self.alignment_forces[alignment]
                for index, r in enumerate(rests):
                    scale = r.scaler(alpha) * r.ramp(timestep)
                    i, j = self._handle_mapping([r.atom_index_1, r.atom_index_2], state)
                    #assert atoms[0] == r.atom_index_1

                    force.setBondParameters(
                        index,
                        [i, j],
                        [
                            r.d_obs,
                            r.kappa,
                            scale * r.force_const,
                            r.tolerance,
                            r.quadratic_cut,
                            r.weight,
                        ],
                    )
                    force.updateParametersInContext(simulation.context)

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

def _create_rdc_force(alignment_index, scale_factor):
    force = mm.CustomCompoundBondForce(
        2,
        f"""
        (1 - step(dev - quadcut)) * quad + step(dev - quadcut) * linear;
        linear = 0.5 * weight * k_rdc * quadcut^2 + weight * k_rdc * quadcut * (dev - quadcut);
        quad = 0.5 * weight * k_rdc * dev^2;
        dev = max(0, abs(d_obs - dcalc) - flat);
        dcalc = pre * (comp1 + comp2 + comp3 + comp4 + comp5);
        pre = kappa_rdc / r^5;
        comp1 = {scale_factor} * rdc_{alignment_index}_s1 * (rx^2 - ry^2);
        comp2 = {scale_factor} * rdc_{alignment_index}_s2 * (3 * rz^2 - r^2);
        comp3 = {scale_factor} * rdc_{alignment_index}_s3 * 2 * rx * ry;
        comp4 = {scale_factor} * rdc_{alignment_index}_s4 * 2 * rx * rz;
        comp5 = {scale_factor} * rdc_{alignment_index}_s5 * 2 * ry * rz;
        rx = x1 - x2;
        ry = y1 - y2;
        rz = z1 - z2;
        r = distance(p1, p2);
        """,
    )
    for i in range(5):
        force.addGlobalParameter(f"rdc_{alignment_index}_s{i + 1}", 0.0)
        force.addEnergyParameterDerivative(f"rdc_{alignment_index}_s{i + 1}")

    force.addPerBondParameter("d_obs")
    force.addPerBondParameter("kappa_rdc")
    force.addPerBondParameter("k_rdc")
    force.addPerBondParameter("flat")
    force.addPerBondParameter("quadcut")
    force.addPerBondParameter("weight")
    force.setForceGroup(FORCE_GROUP)

    return force
