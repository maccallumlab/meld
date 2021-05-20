#
# All rights reserved
#

"""
This module implements transformers that add rdc restraints
"""

import logging

logger = logging.getLogger(__name__)

from meld.runner.transform.restraints.util import _delete_from_always_active
from meld import interfaces
from meld.system import restraints
from meld.system import options
from meld.system import param_sampling
from meld.runner import transform

from simtk import openmm as mm  # type: ignore
from simtk.openmm import app  # type: ignore

from collections import OrderedDict, Callable
from typing import List


class RDCRestraintTransformer(transform.TransformerBase):
    """
    Transformer to handle RDC restraints
    """

    force: mm.CustomCentroidBondForce

    def __init__(
        self,
        param_manager: param_sampling.ParameterManager,
        options: options.RunOptions,
        always_active_restraints: List[restraints.Restraint],
        selectively_active_restraints: List[restraints.SelectivelyActiveCollection],
    ) -> None:
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

        if self.active:
            # map experiments to restraints
            self.expt_dict = DefaultOrderedDict(list)
            for r in self.restraints:
                self.expt_dict[r.expt_index].append(r)

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
                    com_ind.add(r.atom_index_1)
                    com_ind.add(r.atom_index_2)

                # add groups for the COM and dummy particles
                s1 = self.expt_dict[experiment][0].s1_index
                s2 = self.expt_dict[experiment][0].s2_index
                g1 = rdc_force.addGroup(list(com_ind))
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
                    g4 = rdc_force.addGroup([r.atom_index_1])
                    g5 = rdc_force.addGroup([r.atom_index_2])
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

    def update(
        self,
        state: interfaces.IState,
        simulation: app.Simulation,
        alpha: float,
        timestep: int,
    ) -> None:
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
